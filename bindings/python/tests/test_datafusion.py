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

import os
import tempfile

import pyarrow as pa
from datafusion import SessionContext

from pypaimon_rust.datafusion import PaimonCatalog, SQLContext

WAREHOUSE = os.environ.get("PAIMON_TEST_WAREHOUSE", "/tmp/paimon-warehouse")


def extract_rows(batches):
    table = pa.Table.from_batches(batches)
    return sorted(zip(table["id"].to_pylist(), table["name"].to_pylist()))


def test_query_simple_table_via_catalog_provider():
    catalog = PaimonCatalog({"warehouse": WAREHOUSE})
    ctx = SessionContext()
    ctx.register_catalog_provider("paimon", catalog)

    df = ctx.sql("SELECT id, name FROM paimon.default.simple_log_table")

    assert extract_rows(df.collect()) == [
        (1, "alice"),
        (2, "bob"),
        (3, "carol"),
    ]


def test_sql_context_ddl_dml():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})

        ctx.sql("CREATE SCHEMA paimon.test_db")
        ctx.sql(
            "CREATE TABLE paimon.test_db.users "
            "(id INT, name STRING, PRIMARY KEY (id))"
        )

        ctx.sql("INSERT INTO paimon.test_db.users VALUES (1, 'alice'), (2, 'bob')")

        batches = ctx.sql("SELECT id, name FROM paimon.test_db.users")
        table = pa.Table.from_batches(batches)
        rows = sorted(zip(table["id"].to_pylist(), table["name"].to_pylist()))
        assert rows == [(1, "alice"), (2, "bob")]

        ctx.sql("DROP TABLE paimon.test_db.users")
        ctx.sql("DROP SCHEMA paimon.test_db")


def test_register_batch_fully_qualified():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})

        batch = pa.record_batch([[1, 2], ["alice", "bob"]], names=["id", "name"])
        ctx.register_batch("paimon.default.my_temp", batch)

        batches = ctx.sql("SELECT id, name FROM paimon.default.my_temp")
        table = pa.Table.from_batches(batches)
        rows = sorted(zip(table["id"].to_pylist(), table["name"].to_pylist()))
        assert rows == [(1, "alice"), (2, "bob")]

        ctx.sql("DROP TEMPORARY TABLE paimon.default.my_temp")


def test_register_batch_bare_name():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})

        batch = pa.record_batch([[1, 2], ["alice", "bob"]], names=["id", "name"])
        # Bare name uses current catalog and current database
        ctx.register_batch("my_temp", batch)

        batches = ctx.sql("SELECT id, name FROM paimon.default.my_temp")
        table = pa.Table.from_batches(batches)
        rows = sorted(zip(table["id"].to_pylist(), table["name"].to_pylist()))
        assert rows == [(1, "alice"), (2, "bob")]

        ctx.sql("DROP TEMPORARY TABLE paimon.default.my_temp")


def test_register_scalar_function_from_python():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})

        batch = pa.record_batch([[1, None, 3]], names=["id"])
        ctx.register_batch("my_temp", batch)

        def plus_ten(values):
            return pa.array(
                [None if value is None else value + 10 for value in values.to_pylist()],
                type=pa.int64(),
            )

        ctx.register_scalar_function("plus_ten", plus_ten, ["int64"], "int64")

        batches = ctx.sql(
            "SELECT plus_ten(id) AS id FROM paimon.default.my_temp ORDER BY id"
        )
        table = pa.Table.from_batches(batches)
        assert table["id"].to_pylist() == [11, 13, None]

        ctx.sql("DROP TEMPORARY TABLE paimon.default.my_temp")


def test_register_scalar_function_multi_input_plan():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})

        batch = pa.record_batch([[1, 2, 3]], names=["id"])
        ctx.register_batch("my_temp", batch)

        def plus_ten(values):
            return pa.array([value + 10 for value in values.to_pylist()], type=pa.int64())

        ctx.register_scalar_function("plus_ten", plus_ten, ["int64"], "int64")

        batches = ctx.sql(
            """
            SELECT plus_ten(id) AS id FROM paimon.default.my_temp
            UNION ALL
            SELECT plus_ten(id) AS id FROM paimon.default.my_temp
            ORDER BY id
            """
        )
        table = pa.Table.from_batches(batches)
        assert table["id"].to_pylist() == [11, 11, 12, 12, 13, 13]

        ctx.sql("DROP TEMPORARY TABLE paimon.default.my_temp")


def test_register_scalar_function_rejects_non_callable():
    ctx = SQLContext()
    try:
        ctx.register_scalar_function("bad", 1, ["int64"], "int64")
        assert False, "expected non-callable UDF registration to fail"
    except TypeError as e:
        assert "func must be callable" in str(e)


def test_register_scalar_function_rejects_unsupported_type():
    ctx = SQLContext()

    def identity(values):
        return values

    try:
        ctx.register_scalar_function("identity", identity, ["date32"], "date32")
        assert False, "expected unsupported type registration to fail"
    except TypeError as e:
        assert "Unsupported Arrow type" in str(e)


def test_python_scalar_function_exception_surfaces():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})
        ctx.register_batch("my_temp", pa.record_batch([[1]], names=["id"]))

        def boom(values):
            raise RuntimeError("boom")

        ctx.register_scalar_function("boom", boom, ["int64"], "int64")

        try:
            ctx.sql("SELECT boom(id) AS id FROM paimon.default.my_temp")
            assert False, "expected Python UDF exception to fail the query"
        except Exception as e:
            message = str(e)
            assert "Python UDF 'boom' failed" in message
            assert "boom" in message


def test_python_scalar_function_rejects_wrong_length():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})
        ctx.register_batch("my_temp", pa.record_batch([[1, 2]], names=["id"]))

        def wrong_length(values):
            return pa.array([1], type=pa.int64())

        ctx.register_scalar_function("wrong_length", wrong_length, ["int64"], "int64")

        try:
            ctx.sql("SELECT wrong_length(id) AS id FROM paimon.default.my_temp")
            assert False, "expected wrong-length UDF result to fail the query"
        except Exception as e:
            message = str(e)
            assert "Python UDF 'wrong_length' returned 1 rows, expected 2" in message


def test_python_scalar_function_rejects_wrong_type():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})
        ctx.register_batch("my_temp", pa.record_batch([[1]], names=["id"]))

        def wrong_type(values):
            return pa.array(["not an int"], type=pa.string())

        ctx.register_scalar_function("wrong_type", wrong_type, ["int64"], "int64")

        try:
            ctx.sql("SELECT wrong_type(id) AS id FROM paimon.default.my_temp")
            assert False, "expected wrong-type UDF result to fail the query"
        except Exception as e:
            message = str(e)
            assert "Python UDF 'wrong_type' returned Utf8, expected Int64" in message


def test_temp_table_shadows_paimon_table():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})

        ctx.sql("CREATE SCHEMA paimon.test_db")
        ctx.sql("CREATE TABLE paimon.test_db.users (id INT, name STRING)")
        ctx.sql("INSERT INTO paimon.test_db.users VALUES (1, 'real')")

        batch = pa.record_batch([[2], ["temp"]], names=["id", "name"])
        ctx.register_batch("paimon.test_db.users", batch)

        # Temp table should shadow the real Paimon table
        batches = ctx.sql("SELECT id, name FROM paimon.test_db.users")
        table = pa.Table.from_batches(batches)
        rows = sorted(zip(table["id"].to_pylist(), table["name"].to_pylist()))
        assert rows == [(2, "temp")]

        ctx.sql("DROP TEMPORARY TABLE paimon.test_db.users")

        # After dropping, the real table is visible again
        batches = ctx.sql("SELECT id, name FROM paimon.test_db.users")
        table = pa.Table.from_batches(batches)
        rows = sorted(zip(table["id"].to_pylist(), table["name"].to_pylist()))
        assert rows == [(1, "real")]

        ctx.sql("DROP TABLE paimon.test_db.users")
        ctx.sql("DROP SCHEMA paimon.test_db")


def test_drop_temp_table_if_exists():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})

        batch = pa.record_batch([[1]], names=["id"])
        ctx.register_batch("paimon.default.my_temp", batch)

        ctx.sql("DROP TEMPORARY TABLE IF EXISTS paimon.default.my_temp")

        # Should be able to drop again without error
        ctx.sql("DROP TEMPORARY TABLE IF EXISTS paimon.default.my_temp")


def test_multi_catalog_temp_table():
    with tempfile.TemporaryDirectory() as wh1, tempfile.TemporaryDirectory() as wh2:
        ctx = SQLContext()
        ctx.register_catalog("cat1", {"warehouse": wh1})
        ctx.register_catalog("cat2", {"warehouse": wh2})

        batch1 = pa.record_batch([[1]], names=["id"])
        batch2 = pa.record_batch([[2]], names=["id"])

        ctx.register_batch("cat1.default.t1", batch1)
        ctx.register_batch("cat2.default.t2", batch2)

        result1 = ctx.sql("SELECT id FROM cat1.default.t1")
        assert pa.Table.from_batches(result1)["id"].to_pylist() == [1]

        result2 = ctx.sql("SELECT id FROM cat2.default.t2")
        assert pa.Table.from_batches(result2)["id"].to_pylist() == [2]

        ctx.sql("DROP TEMPORARY TABLE cat1.default.t1")
        ctx.sql("DROP TEMPORARY TABLE cat2.default.t2")


def test_register_batch_invalid_catalog():
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})

        batch = pa.record_batch([[1]], names=["id"])
        try:
            ctx.register_batch("unknown_catalog.default.my_temp", batch)
            assert False, "Expected an error for unknown catalog"
        except Exception as e:
            assert "unknown_catalog" in str(e).lower() or "not a paimon" in str(e).lower() or "unknown" in str(e).lower()


def test_table_functions_registered_with_catalog():
    """register_catalog auto-registers vector_search / full_text_search as
    UDTFs. Calling one with the wrong argument count surfaces the function's
    own validation error, which proves it is registered — an unregistered
    name would instead fail with 'table function not found'."""
    with tempfile.TemporaryDirectory() as warehouse:
        ctx = SQLContext()
        ctx.register_catalog("paimon", {"warehouse": warehouse})

        for fn in ("vector_search", "full_text_search"):
            try:
                ctx.sql(f"SELECT * FROM {fn}('only_one_arg')")
                assert False, f"expected {fn} to reject a single argument"
            except Exception as e:
                assert "requires 4 arguments" in str(e), str(e)
