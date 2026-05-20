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

use std::any::Any;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{make_array, Array, ArrayData, ArrayRef};
use arrow::datatypes::DataType as ArrowDataType;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use datafusion::catalog::CatalogProvider;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_ffi::catalog_provider::FFI_CatalogProvider;
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use paimon::{CatalogFactory, Options};
use paimon_datafusion::{PaimonCatalogProvider, SQLContext};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

use crate::error::{df_to_py_err, to_py_err};
use paimon_datafusion::runtime::runtime;

fn build_paimon_catalog_provider(
    catalog_options: HashMap<String, String>,
) -> PyResult<Arc<PaimonCatalogProvider>> {
    let rt = runtime();
    rt.block_on(async {
        let options = Options::from_map(catalog_options);
        let catalog = CatalogFactory::create(options).await.map_err(to_py_err)?;
        Ok::<_, PyErr>(Arc::new(PaimonCatalogProvider::new(catalog)))
    })
}

fn ffi_logical_codec_from_pycapsule(obj: Bound<'_, PyAny>) -> PyResult<FFI_LogicalExtensionCodec> {
    let attr_name = "__datafusion_logical_extension_codec__";
    let capsule = if obj.hasattr(attr_name)? {
        obj.getattr(attr_name)?.call0()?
    } else {
        obj
    };

    let capsule = capsule.cast::<PyCapsule>()?;
    let expected_name = c"datafusion_logical_extension_codec";
    let ptr = capsule.pointer_checked(Some(expected_name))?;
    let codec = unsafe { ptr.cast::<FFI_LogicalExtensionCodec>().as_ref() };

    Ok(codec.clone())
}

fn parse_arrow_type(type_name: &str) -> PyResult<ArrowDataType> {
    match type_name.to_ascii_lowercase().as_str() {
        "bool" | "boolean" => Ok(ArrowDataType::Boolean),
        "int8" => Ok(ArrowDataType::Int8),
        "int16" => Ok(ArrowDataType::Int16),
        "int" | "int32" | "integer" => Ok(ArrowDataType::Int32),
        "bigint" | "int64" | "long" => Ok(ArrowDataType::Int64),
        "float" | "float32" => Ok(ArrowDataType::Float32),
        "double" | "float64" => Ok(ArrowDataType::Float64),
        "string" | "utf8" => Ok(ArrowDataType::Utf8),
        "large_string" | "large_utf8" => Ok(ArrowDataType::LargeUtf8),
        "binary" => Ok(ArrowDataType::Binary),
        "large_binary" => Ok(ArrowDataType::LargeBinary),
        other => Err(PyTypeError::new_err(format!(
            "Unsupported Arrow type for Python UDF: {other}"
        ))),
    }
}

fn df_execution_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Execution(message.into())
}

fn columnar_value_to_array(value: &ColumnarValue, num_rows: usize) -> DFResult<ArrayRef> {
    match value {
        ColumnarValue::Array(array) => Ok(Arc::clone(array)),
        ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(num_rows),
    }
}

struct PyScalarUDF {
    name: String,
    func: Py<PyAny>,
    input_types: Vec<ArrowDataType>,
    return_type: ArrowDataType,
    signature: Signature,
}

impl PyScalarUDF {
    fn new(
        name: String,
        func: Py<PyAny>,
        input_types: Vec<ArrowDataType>,
        return_type: ArrowDataType,
    ) -> Self {
        let signature = Signature::exact(input_types.clone(), Volatility::Volatile);
        Self {
            name,
            func,
            input_types,
            return_type,
            signature,
        }
    }
}

impl Debug for PyScalarUDF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyScalarUDF")
            .field("name", &self.name)
            .field("input_types", &self.input_types)
            .field("return_type", &self.return_type)
            .finish_non_exhaustive()
    }
}

impl PartialEq for PyScalarUDF {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.input_types == other.input_types
            && self.return_type == other.return_type
    }
}

impl Eq for PyScalarUDF {}

impl Hash for PyScalarUDF {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.input_types.hash(state);
        self.return_type.hash(state);
    }
}

impl ScalarUDFImpl for PyScalarUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[ArrowDataType]) -> DFResult<ArrowDataType> {
        Ok(self.return_type.clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let arrays = args
            .args
            .iter()
            .map(|value| columnar_value_to_array(value, args.number_rows))
            .collect::<DFResult<Vec<_>>>()?;

        let output = Python::try_attach(|py| -> PyResult<ArrayRef> {
            let py_args = arrays
                .iter()
                .map(|array| array.to_data().to_pyarrow(py))
                .collect::<PyResult<Vec<_>>>()?;
            let py_args = PyTuple::new(py, py_args)?;
            let output = self.func.bind(py).call1(py_args)?;
            Ok(make_array(ArrayData::from_pyarrow_bound(&output)?))
        })
        .ok_or_else(|| df_execution_error("Python interpreter is not available"))?
        .map_err(|err| df_execution_error(format!("Python UDF '{}' failed: {err}", self.name)))?;

        if output.len() != args.number_rows {
            return Err(df_execution_error(format!(
                "Python UDF '{}' returned {} rows, expected {}",
                self.name,
                output.len(),
                args.number_rows
            )));
        }
        if output.data_type() != &self.return_type {
            return Err(df_execution_error(format!(
                "Python UDF '{}' returned {:?}, expected {:?}",
                self.name,
                output.data_type(),
                self.return_type
            )));
        }

        Ok(ColumnarValue::Array(output))
    }
}

/// A Paimon catalog exportable to Python DataFusion `SessionContext`.
#[pyclass(name = "PaimonCatalog")]
pub struct PaimonCatalog {
    provider: Arc<PaimonCatalogProvider>,
}

#[pymethods]
impl PaimonCatalog {
    /// Create a Paimon catalog that can be registered into a DataFusion session.
    #[new]
    fn new(catalog_options: HashMap<String, String>) -> PyResult<Self> {
        Ok(Self {
            provider: build_paimon_catalog_provider(catalog_options)?,
        })
    }

    /// Export this catalog as a DataFusion catalog provider PyCapsule.
    fn __datafusion_catalog_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let name = cr"datafusion_catalog_provider".into();
        let provider = Arc::clone(&self.provider) as Arc<dyn CatalogProvider + Send>;
        let codec = ffi_logical_codec_from_pycapsule(session)?;
        let provider = FFI_CatalogProvider::new_with_ffi_codec(provider, Some(runtime()), codec);
        PyCapsule::new(py, provider, Some(name))
    }
}

/// A SQL context that supports registering multiple Paimon catalogs and executing SQL.
#[pyclass(name = "SQLContext")]
pub struct PySQLContext {
    inner: SQLContext,
}

#[pymethods]
impl PySQLContext {
    #[new]
    fn new() -> Self {
        Self {
            inner: SQLContext::new(),
        }
    }

    fn register_catalog(
        &mut self,
        catalog_name: String,
        catalog_options: HashMap<String, String>,
    ) -> PyResult<()> {
        let rt = runtime();
        rt.block_on(async {
            let options = Options::from_map(catalog_options);
            let catalog = CatalogFactory::create(options).await.map_err(to_py_err)?;
            self.inner
                .register_catalog(catalog_name, catalog)
                .await
                .map_err(df_to_py_err)
        })
    }

    fn set_current_catalog(&mut self, catalog_name: String) -> PyResult<()> {
        let rt = runtime();
        rt.block_on(async {
            self.inner
                .set_current_catalog(catalog_name)
                .await
                .map_err(df_to_py_err)
        })
    }

    fn set_current_database(&self, database_name: String) -> PyResult<()> {
        let rt = runtime();
        rt.block_on(async {
            self.inner
                .set_current_database(&database_name)
                .await
                .map_err(df_to_py_err)
        })
    }

    fn register_batch(&self, name: String, batch: Bound<'_, PyAny>) -> PyResult<()> {
        let batch = datafusion::arrow::record_batch::RecordBatch::from_pyarrow_bound(&batch)?;
        let schema = batch.schema();
        let mem_table = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]])
            .map_err(df_to_py_err)?;
        self.inner
            .register_temp_table(&name, Arc::new(mem_table))
            .map_err(df_to_py_err)
    }

    fn register_scalar_function(
        &self,
        py: Python<'_>,
        name: String,
        func: Py<PyAny>,
        input_types: Vec<String>,
        return_type: String,
    ) -> PyResult<()> {
        if !func.bind(py).is_callable() {
            return Err(PyTypeError::new_err("func must be callable"));
        }

        let input_types = input_types
            .iter()
            .map(|type_name| parse_arrow_type(type_name))
            .collect::<PyResult<Vec<_>>>()?;
        let return_type = parse_arrow_type(&return_type)?;
        let udf = PyScalarUDF::new(name, func, input_types, return_type);
        self.inner.ctx().register_udf(ScalarUDF::new_from_impl(udf));
        Ok(())
    }

    fn sql(&self, py: Python<'_>, sql: String) -> PyResult<Vec<Py<PyAny>>> {
        let rt = runtime();
        let batches = rt.block_on(async {
            let df = self.inner.sql(&sql).await.map_err(df_to_py_err)?;
            df.collect().await.map_err(df_to_py_err)
        })?;
        batches
            .iter()
            .map(|batch| Ok(batch.to_pyarrow(py)?.unbind()))
            .collect()
    }
}

pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let this = PyModule::new(py, "datafusion")?;
    this.add_class::<PaimonCatalog>()?;
    this.add_class::<PySQLContext>()?;
    m.add_submodule(&this)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pypaimon_rust.datafusion", this)?;
    Ok(())
}
