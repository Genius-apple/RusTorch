use rustorch_core::Tensor;

#[test]
fn test_add_backward() {
    let a = Tensor::new(&[2.0], &[1]).set_requires_grad(true);
    let b = Tensor::new(&[3.0], &[1]).set_requires_grad(true);
    
    let c = &a + &b;
    
    c.backward();
    
    // dc/da = 1, dc/db = 1
    let grad_a = a.grad().expect("grad_a missing");
    let grad_b = b.grad().expect("grad_b missing");
    
    assert_eq!(grad_a.data()[0], 1.0);
    assert_eq!(grad_b.data()[0], 1.0);
}

#[test]
fn test_add_chain() {
    let a = Tensor::new(&[2.0], &[1]).set_requires_grad(true);
    let b = Tensor::new(&[3.0], &[1]).set_requires_grad(true);
    
    let c = &a + &b;
    let d = &c + &a; // d = (a+b) + a = 2a + b
    
    d.backward();
    
    // dd/da = 2, dd/db = 1
    let grad_a = a.grad().expect("grad_a missing");
    let grad_b = b.grad().expect("grad_b missing");
    
    assert_eq!(grad_a.data()[0], 2.0);
    assert_eq!(grad_b.data()[0], 1.0);
}
