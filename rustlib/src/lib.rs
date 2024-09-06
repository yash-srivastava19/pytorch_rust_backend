// This is for wrapping things up. 
use pyo3::prelude::* ;
use pyo3::wrap_pyfunction;

// Here come our library stuff
mod value;
pub use crate::value::Value ;

mod neuron;
pub use crate::neuron::Neuron ;

mod layer;
pub use crate::layer::Layer ;

mod mlp;
pub use crate::mlp::MLP ;

// There are some ways to do what we want to do, make the pyfunctions, and then 

#[pyfunction]
fn scaler_backprop(val1: f64, val2: f64, weight1: f64, weight2: f64, bias:f64) {
    let x1 = Value::from(val1).with_label("x1") ;
    let x1_clone = x1.clone();

    let x2 = Value::from(val2).with_label("x2") ;

    let w1 = Value::from(weight1).with_label("w1");
    let w2 = Value::from(weight2).with_label("w2");

    let b = Value::from(bias).with_label("b");

    let x1w1 = (x1*w1).with_label("x1w1");
    let x2w2 = (x2*w2).with_label("x2w2");

    let x1w1x2w2 = (x1w1 + x2w2).with_label("x1w1x2w2");

    let n = (x1w1x2w2 + b).with_label("n");
    let o = n.relu().with_label("o");

    o.backwards() ;

    println!("O Data {}", o.data());
    println!("X1 Gradient {}", x1_clone.gradient());

}

#[pymodule]
fn rustlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scaler_backprop, m)?)? ;
    Ok(())
}
