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

//! Consumer progress manager for Java-compatible consumer files.

use std::time::Duration;

use crate::io::FileIO;
use serde::Deserialize;

const CONSUMER_DIR: &str = "consumer";
const CONSUMER_PREFIX: &str = "consumer-";
const READ_RETRIES: usize = 10;
const RETRY_DELAY: Duration = Duration::from_millis(200);

/// Reads consumer progress files stored under a table or branch.
#[derive(Debug, Clone)]
pub struct ConsumerManager {
    file_io: FileIO,
    table_path: String,
}

impl ConsumerManager {
    pub fn new(file_io: FileIO, table_path: String) -> Self {
        Self {
            file_io,
            table_path,
        }
    }

    /// Create a manager for a branch of this table.
    pub fn with_branch(&self, branch_name: &str) -> Self {
        Self::new(
            self.file_io.clone(),
            format!("{}/branch/branch-{branch_name}", self.table_path),
        )
    }

    /// Read one consumer's next snapshot id.
    pub async fn get(&self, consumer_id: &str) -> crate::Result<Option<i64>> {
        // A listed consumer id is a single file name. Do not let a pushed SQL
        // literal turn the direct-read optimization into path traversal.
        if consumer_id.contains('/') {
            return Ok(None);
        }

        let path = format!(
            "{}/{}/{}{}",
            self.table_path, CONSUMER_DIR, CONSUMER_PREFIX, consumer_id
        );
        let input = self.file_io.new_input(&path)?;
        for attempt in 0..READ_RETRIES {
            let bytes = match input.read().await {
                Ok(bytes) => bytes,
                Err(crate::Error::IoUnexpected { ref source, .. })
                    if source.kind() == opendal::ErrorKind::NotFound =>
                {
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            match serde_json::from_slice::<Consumer>(&bytes) {
                Ok(consumer) => return Ok(Some(consumer.next_snapshot)),
                Err(_) if attempt + 1 < READ_RETRIES => tokio::time::sleep(RETRY_DELAY).await,
                Err(error) => {
                    return Err(crate::Error::DataInvalid {
                        message: format!("consumer '{consumer_id}' JSON invalid: {error}"),
                        source: Some(Box::new(error)),
                    });
                }
            }
        }
        unreachable!("READ_RETRIES is non-zero")
    }

    /// List consumer ids and their next snapshot ids, sorted by consumer id.
    pub async fn list_all(&self) -> crate::Result<Vec<(String, i64)>> {
        let directory = format!("{}/{}", self.table_path, CONSUMER_DIR);
        let statuses = match self.file_io.list_status(&directory).await {
            Ok(statuses) => statuses,
            Err(crate::Error::IoUnexpected { ref source, .. })
                if source.kind() == opendal::ErrorKind::NotFound =>
            {
                return Ok(Vec::new());
            }
            Err(error) => return Err(error),
        };

        let mut consumers = Vec::new();
        for status in statuses {
            if status.is_dir {
                continue;
            }
            let name = status.path.rsplit('/').next().unwrap_or(&status.path);
            let Some(id) = name.strip_prefix(CONSUMER_PREFIX) else {
                continue;
            };
            if let Some(next_snapshot) = self.get(id).await? {
                consumers.push((id.to_string(), next_snapshot));
            }
        }
        consumers.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok(consumers)
    }
}

#[derive(Deserialize)]
struct Consumer {
    #[serde(rename = "nextSnapshot")]
    next_snapshot: i64,
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;
    use crate::io::FileIOBuilder;

    #[tokio::test]
    async fn retries_a_consumer_being_overwritten() {
        let file_io = FileIOBuilder::new("memory").build().unwrap();
        let table_path = "memory:/consumer-retry";
        let directory = format!("{table_path}/{CONSUMER_DIR}");
        let path = format!("{directory}/{CONSUMER_PREFIX}job");
        file_io.mkdirs(&directory).await.unwrap();
        file_io
            .new_output(&path)
            .unwrap()
            .write(Bytes::from_static(b"{"))
            .await
            .unwrap();

        let manager = ConsumerManager::new(file_io.clone(), table_path.to_string());
        let mut read = Box::pin(manager.get("job"));
        tokio::select! {
            result = &mut read => panic!("invalid JSON returned before retry: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
        file_io
            .new_output(&path)
            .unwrap()
            .write(Bytes::from_static(br#"{"nextSnapshot":5}"#))
            .await
            .unwrap();

        assert_eq!(read.await.unwrap(), Some(5));
    }
}
