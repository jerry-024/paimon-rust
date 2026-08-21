// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const BYTE_PERMIT_UNIT: u64 = 1024 * 1024;
const DEFAULT_PARALLELISM: usize = 8;
const DEFAULT_MAX_INFLIGHT_BYTES: u64 = 256 * 1024 * 1024;
/// A single row group's byte accounting is capped to the budget divided by
/// this share, so one oversized row group (e.g. a ~294 MiB projected wide
/// vector column under the 256 MiB default budget) cannot consume every
/// byte permit and silently serialize the scan. This makes the byte budget
/// a fair-admission mechanism for large row groups rather than a strict
/// projected-byte ceiling: up to `min(parallelism, MAX_BUDGET_SHARES)` such
/// row groups may be in flight, each accounted at `budget / shares` even
/// though its projected size is larger. Capped at 4 shares until wider RSS
/// measurements justify the full parallelism.
const MAX_BUDGET_SHARES: usize = 4;

/// Shared resource budget for concurrent Parquet row-group reads.
#[derive(Debug)]
pub struct ParquetReadBudget {
    parallelism: usize,
    row_groups: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
    byte_permits: u32,
    diagnostics: Arc<ParquetReadDiagnostics>,
}

#[derive(Debug)]
struct ParquetReadDiagnostics {
    enabled: AtomicBool,
    row_group_count: AtomicU64,
    projected_bytes_min: AtomicU64,
    projected_bytes_max: AtomicU64,
    projected_bytes_total: AtomicU64,
    current_inflight: AtomicUsize,
    peak_inflight: AtomicUsize,
}

impl Default for ParquetReadDiagnostics {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            row_group_count: AtomicU64::new(0),
            projected_bytes_min: AtomicU64::new(u64::MAX),
            projected_bytes_max: AtomicU64::new(0),
            projected_bytes_total: AtomicU64::new(0),
            current_inflight: AtomicUsize::new(0),
            peak_inflight: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ParquetReadDiagnosticsSnapshot {
    pub(crate) row_group_count: u64,
    pub(crate) projected_bytes_min: u64,
    pub(crate) projected_bytes_max: u64,
    pub(crate) projected_bytes_total: u64,
    pub(crate) current_inflight: usize,
    pub(crate) peak_inflight: usize,
}

impl ParquetReadBudget {
    pub fn new(parallelism: usize, max_inflight_bytes: u64) -> crate::Result<Self> {
        if parallelism == 0 || parallelism > Semaphore::MAX_PERMITS {
            return Err(crate::Error::DataInvalid {
                message: format!(
                    "Parquet row-group parallelism must be between 1 and {}, got {parallelism}",
                    Semaphore::MAX_PERMITS
                ),
                source: None,
            });
        }
        if max_inflight_bytes == 0 {
            return Err(crate::Error::DataInvalid {
                message: "Parquet row-group max in-flight bytes must be greater than 0".to_string(),
                source: None,
            });
        }
        let max_byte_permits = Semaphore::MAX_PERMITS.min(u32::MAX as usize) as u32;
        let byte_permits = max_inflight_bytes
            .div_ceil(BYTE_PERMIT_UNIT)
            .min(u64::from(max_byte_permits)) as u32;

        Ok(Self {
            parallelism,
            row_groups: Arc::new(Semaphore::new(parallelism)),
            bytes: Arc::new(Semaphore::new(byte_permits as usize)),
            byte_permits,
            diagnostics: Arc::new(ParquetReadDiagnostics::default()),
        })
    }

    pub fn parallelism(&self) -> usize {
        self.parallelism
    }

    pub(crate) fn enable_diagnostics(&self) {
        self.diagnostics.enabled.store(true, Ordering::Relaxed);
    }

    pub(crate) fn diagnostics_enabled(&self) -> bool {
        self.diagnostics.enabled.load(Ordering::Relaxed)
    }

    pub(crate) fn record_projected_row_groups(&self, projected_bytes: &[u64]) {
        if !self.diagnostics_enabled() || projected_bytes.is_empty() {
            return;
        }
        self.diagnostics
            .row_group_count
            .fetch_add(projected_bytes.len() as u64, Ordering::Relaxed);
        self.diagnostics.projected_bytes_min.fetch_min(
            *projected_bytes.iter().min().expect("checked non-empty"),
            Ordering::Relaxed,
        );
        self.diagnostics.projected_bytes_max.fetch_max(
            *projected_bytes.iter().max().expect("checked non-empty"),
            Ordering::Relaxed,
        );
        self.diagnostics.projected_bytes_total.fetch_add(
            projected_bytes
                .iter()
                .copied()
                .fold(0u64, u64::saturating_add),
            Ordering::Relaxed,
        );
    }

    pub(crate) fn diagnostics(&self) -> ParquetReadDiagnosticsSnapshot {
        let row_group_count = self.diagnostics.row_group_count.load(Ordering::Relaxed);
        ParquetReadDiagnosticsSnapshot {
            row_group_count,
            projected_bytes_min: if row_group_count == 0 {
                0
            } else {
                self.diagnostics.projected_bytes_min.load(Ordering::Relaxed)
            },
            projected_bytes_max: self.diagnostics.projected_bytes_max.load(Ordering::Relaxed),
            projected_bytes_total: self
                .diagnostics
                .projected_bytes_total
                .load(Ordering::Relaxed),
            current_inflight: self.diagnostics.current_inflight.load(Ordering::Relaxed),
            peak_inflight: self.diagnostics.peak_inflight.load(Ordering::Relaxed),
        }
    }

    pub(crate) async fn acquire(
        &self,
        projected_uncompressed_bytes: u64,
    ) -> crate::Result<ParquetReadPermit> {
        let row_group = Arc::clone(&self.row_groups)
            .acquire_owned()
            .await
            .map_err(|error| crate::Error::UnexpectedError {
                message: "Parquet row-group read budget was closed".to_string(),
                source: Some(Box::new(error)),
            })?;
        // Cap a single row group's accounting to a fair share of the budget
        // (see MAX_BUDGET_SHARES): an oversized row group must not take every
        // permit and serialize the scan. `parallelism >= 1` is validated at
        // construction, and `max(1)` keeps tiny budgets sound — the semaphore
        // still bounds total in-flight permits.
        let shares = self.parallelism.min(MAX_BUDGET_SHARES) as u64;
        let share_cap = u64::from(self.byte_permits).div_ceil(shares).max(1);
        let requested = projected_uncompressed_bytes
            .max(1)
            .div_ceil(BYTE_PERMIT_UNIT)
            .min(share_cap)
            .min(u64::from(self.byte_permits)) as u32;
        let bytes = Arc::clone(&self.bytes)
            .acquire_many_owned(requested)
            .await
            .map_err(|error| crate::Error::UnexpectedError {
                message: "Parquet byte read budget was closed".to_string(),
                source: Some(Box::new(error)),
            })?;
        let diagnostics = self.diagnostics_enabled().then(|| {
            let current = self
                .diagnostics
                .current_inflight
                .fetch_add(1, Ordering::Relaxed)
                + 1;
            self.diagnostics
                .peak_inflight
                .fetch_max(current, Ordering::Relaxed);
            Arc::clone(&self.diagnostics)
        });
        Ok(ParquetReadPermit {
            _row_group: row_group,
            _bytes: bytes,
            diagnostics,
        })
    }
}

impl Default for ParquetReadBudget {
    fn default() -> Self {
        Self::new(DEFAULT_PARALLELISM, DEFAULT_MAX_INFLIGHT_BYTES)
            .expect("default Parquet read budget is valid")
    }
}

#[derive(Debug)]
pub(crate) struct ParquetReadPermit {
    _row_group: OwnedSemaphorePermit,
    _bytes: OwnedSemaphorePermit,
    diagnostics: Option<Arc<ParquetReadDiagnostics>>,
}

impl Drop for ParquetReadPermit {
    fn drop(&mut self) {
        if let Some(diagnostics) = &self.diagnostics {
            diagnostics.current_inflight.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn shared_budget_blocks_until_permits_are_released() {
        let budget = Arc::new(ParquetReadBudget::new(2, BYTE_PERMIT_UNIT).unwrap());
        let first = budget.acquire(2 * BYTE_PERMIT_UNIT).await.unwrap();

        assert!(
            tokio::time::timeout(Duration::from_millis(20), budget.acquire(1))
                .await
                .is_err(),
            "the projected-byte budget must be shared across readers"
        );

        drop(first);
        tokio::time::timeout(Duration::from_secs(1), budget.acquire(1))
            .await
            .expect("dropping a read must release its permits")
            .unwrap();
    }

    #[tokio::test]
    async fn diagnostics_aggregate_shared_row_group_reads() {
        let budget = Arc::new(ParquetReadBudget::new(2, 2 * BYTE_PERMIT_UNIT).unwrap());
        budget.enable_diagnostics();
        budget.record_projected_row_groups(&[300, 100, 200]);

        let first = budget.acquire(1).await.unwrap();
        let second = budget.acquire(1).await.unwrap();
        assert_eq!(
            budget.diagnostics(),
            ParquetReadDiagnosticsSnapshot {
                row_group_count: 3,
                projected_bytes_min: 100,
                projected_bytes_max: 300,
                projected_bytes_total: 600,
                current_inflight: 2,
                peak_inflight: 2,
            }
        );

        drop(first);
        drop(second);
        assert_eq!(budget.diagnostics().current_inflight, 0);
        assert_eq!(budget.diagnostics().peak_inflight, 2);
    }

    #[test]
    fn rejects_invalid_limits() {
        assert!(ParquetReadBudget::new(0, BYTE_PERMIT_UNIT).is_err());
        assert!(ParquetReadBudget::new(1, 0).is_err());
        assert!(
            ParquetReadBudget::new(Semaphore::MAX_PERMITS.saturating_add(1), BYTE_PERMIT_UNIT)
                .is_err()
        );
    }

    /// A row group larger than the whole budget must not serialize the scan:
    /// its accounting is capped to budget / min(parallelism, MAX_BUDGET_SHARES),
    /// so min(parallelism, MAX_BUDGET_SHARES) oversized row groups run
    /// concurrently. This is the wide-vector-column case (a ~294 MiB projected
    /// row group under the 256 MiB default budget).
    #[tokio::test]
    async fn oversized_row_groups_share_the_budget() {
        // parallelism 8 > MAX_BUDGET_SHARES: shares = 4.
        let budget = Arc::new(ParquetReadBudget::new(8, 8 * BYTE_PERMIT_UNIT).unwrap());
        budget.enable_diagnostics();
        let oversized = 100 * BYTE_PERMIT_UNIT; // far above the whole budget

        // 4 oversized acquisitions must all succeed (each accounted at 2 permits).
        let mut permits = Vec::new();
        for _ in 0..MAX_BUDGET_SHARES {
            permits.push(
                tokio::time::timeout(Duration::from_secs(1), budget.acquire(oversized))
                    .await
                    .expect("an oversized row group must only take a fair share")
                    .unwrap(),
            );
        }
        assert_eq!(budget.diagnostics().peak_inflight, MAX_BUDGET_SHARES);

        // The 5th oversized request exhausts the byte semaphore and must wait.
        assert!(
            tokio::time::timeout(Duration::from_millis(20), budget.acquire(oversized))
                .await
                .is_err(),
            "the byte budget still bounds total in-flight accounting"
        );

        drop(permits);
        tokio::time::timeout(Duration::from_secs(1), budget.acquire(oversized))
            .await
            .expect("released shares must become available again")
            .unwrap();
    }

    #[tokio::test]
    async fn non_divisible_budget_does_not_exceed_max_shares() {
        let budget = Arc::new(ParquetReadBudget::new(8, 10 * BYTE_PERMIT_UNIT).unwrap());
        let oversized = 100 * BYTE_PERMIT_UNIT;

        // ceil(10 / 4) = 3 permits, so only 3 oversized reads fit. Rounding
        // down to 2 would incorrectly admit 5 reads and exceed the cap.
        let permits = futures::future::try_join_all((0..3).map(|_| budget.acquire(oversized)))
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), budget.acquire(oversized))
                .await
                .is_err()
        );
        drop(permits);
    }

    /// Row groups at or below budget / shares keep their exact accounting —
    /// the share cap must not change behavior for ordinary layouts.
    #[tokio::test]
    async fn small_row_groups_keep_exact_accounting() {
        let budget = Arc::new(ParquetReadBudget::new(4, 4 * BYTE_PERMIT_UNIT).unwrap());
        // share_cap = 4 / min(4,4) = 1 permit; a 1 MiB row group requests
        // exactly 1 permit, so 4 fit and the 5th blocks — identical to the
        // pre-cap behavior for small row groups.
        let mut permits = Vec::new();
        for _ in 0..4 {
            permits.push(budget.acquire(BYTE_PERMIT_UNIT).await.unwrap());
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(20), budget.acquire(BYTE_PERMIT_UNIT))
                .await
                .is_err()
        );
    }

    /// Tiny budgets stay sound: the cap never rounds a request down to zero
    /// permits, and a budget smaller than the share divisor still admits one
    /// row group at a time.
    #[tokio::test]
    async fn tiny_budget_still_admits_one_at_a_time() {
        let budget = Arc::new(ParquetReadBudget::new(8, BYTE_PERMIT_UNIT).unwrap());
        let first = budget.acquire(100 * BYTE_PERMIT_UNIT).await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), budget.acquire(1))
                .await
                .is_err(),
            "a single-permit budget admits exactly one read"
        );
        drop(first);
        budget.acquire(1).await.unwrap();
    }
}
