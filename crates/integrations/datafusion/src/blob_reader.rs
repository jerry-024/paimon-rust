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

use std::sync::{Arc, RwLock};

use paimon::io::FileIO;

#[derive(Clone, Debug)]
struct BlobFileIO {
    prefix: String,
    file_io: FileIO,
}

/// Session-scoped registry of Paimon [`FileIO`] instances for BlobDescriptor reads.
#[derive(Clone, Debug, Default)]
pub struct BlobReaderRegistry {
    readers: Arc<RwLock<Vec<BlobFileIO>>>,
}

impl BlobReaderRegistry {
    pub fn register(&self, prefix: impl Into<String>, file_io: FileIO) {
        let prefix = prefix.into();
        let mut readers = self.readers.write().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = readers.iter_mut().find(|reader| reader.prefix == prefix) {
            existing.file_io = file_io;
            return;
        }
        readers.push(BlobFileIO { prefix, file_io });
    }

    pub fn register_if_absent(&self, prefix: impl Into<String>, file_io: FileIO) {
        let prefix = prefix.into();
        let mut readers = self.readers.write().unwrap_or_else(|e| e.into_inner());
        if readers.iter().any(|reader| reader.prefix == prefix) {
            return;
        }
        readers.push(BlobFileIO { prefix, file_io });
    }

    pub fn resolve(&self, uri: &str) -> Option<FileIO> {
        let readers = self.readers.read().unwrap_or_else(|e| e.into_inner());
        readers
            .iter()
            .filter(|reader| uri.starts_with(&reader.prefix))
            .max_by_key(|reader| reader.prefix.len())
            .map(|reader| reader.file_io.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use paimon::io::{FileIOBuilder, FileRead};
    use paimon::spec::BlobDescriptor;

    use super::*;

    #[tokio::test]
    async fn resolves_file_blob_descriptor_with_file_io() {
        let directory = tempfile::tempdir().unwrap();
        let blob_path = directory.path().join("blob.bin");
        fs::write(&blob_path, b"prefixpayloadsuffix").unwrap();

        let descriptor = BlobDescriptor::new(
            blob_path.to_string_lossy().to_string(),
            6,
            "payload".len() as i64,
        );
        let descriptor = BlobDescriptor::deserialize(&descriptor.serialize()).unwrap();

        let registry = BlobReaderRegistry::default();
        let file_io = FileIOBuilder::new("file").build().unwrap();
        registry.register(directory.path().to_string_lossy().to_string(), file_io);

        let resolved_file_io = registry
            .resolve(descriptor.uri())
            .expect("file blob descriptor should resolve to registered FileIO");
        let input = resolved_file_io.new_input(descriptor.uri()).unwrap();
        let reader = input.reader().await.unwrap();
        let start = descriptor.offset() as u64;
        let end = start + descriptor.length() as u64;
        let bytes = reader.read(start..end).await.unwrap();

        assert_eq!(&bytes[..], b"payload");
    }
}
