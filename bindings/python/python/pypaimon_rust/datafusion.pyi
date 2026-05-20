# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from typing import Any, Callable, Dict, List

import pyarrow

class PaimonCatalog:
    def __init__(self, catalog_options: Dict[str, str]) -> None: ...
    def __datafusion_catalog_provider__(self, session: Any) -> object: ...

class SQLContext:
    def __init__(self) -> None: ...
    def register_catalog(
        self, catalog_name: str, catalog_options: Dict[str, str]
    ) -> None: ...
    def set_current_catalog(self, catalog_name: str) -> None: ...
    def set_current_database(self, database_name: str) -> None: ...
    def register_batch(self, name: str, batch: pyarrow.RecordBatch) -> None: ...
    def register_scalar_function(
        self,
        name: str,
        func: Callable[..., pyarrow.Array],
        input_types: List[str],
        return_type: str,
    ) -> None:
        """
        Register a Python scalar UDF.

        The callable receives one PyArrow Array per argument and must return a
        PyArrow Array with the declared return type and the same row count.
        Supported type names are: boolean, int8, int16, int32, int64,
        float32, float64, string, large_string, binary, and large_binary.
        Aliases such as bool, int, bigint, long, float, double, utf8,
        large_utf8 are also accepted.
        """
        ...
    def sql(self, sql: str) -> List[pyarrow.RecordBatch]: ...
