use rustorch_core::ops;
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

#[test]
fn test_matmul_forward_precision_under_1e6() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = a.matmul(&b);
    let got = c.data();
    let expected = [19.0f32, 22.0, 43.0, 50.0];
    for i in 0..expected.len() {
        let err = (got[i] - expected[i]).abs();
        assert!(
            err < 1e-6,
            "index {} err {} got {} expected {}",
            i,
            err,
            got[i],
            expected[i]
        );
    }
}

#[test]
fn test_matmul_backward_precision_under_1e6() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).set_requires_grad(true);
    let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).set_requires_grad(true);
    let out = a.matmul(&b);
    let loss = ops::sum(&out);
    loss.backward();

    let grad_a = a.grad().expect("grad_a missing");
    let grad_b = b.grad().expect("grad_b missing");
    let ga = grad_a.data();
    let gb = grad_b.data();

    let expected_a = [11.0f32, 15.0, 11.0, 15.0];
    let expected_b = [4.0f32, 4.0, 6.0, 6.0];

    for i in 0..expected_a.len() {
        let err = (ga[i] - expected_a[i]).abs();
        assert!(err < 1e-6, "grad_a idx {} err {}", i, err);
    }
    for i in 0..expected_b.len() {
        let err = (gb[i] - expected_b[i]).abs();
        assert!(err < 1e-6, "grad_b idx {} err {}", i, err);
    }
}

#[test]
fn test_conv2d_forward_precision_under_1e6() {
    let input = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    );
    let weight = Tensor::new(&[1.0, 0.0, 0.0, -1.0], &[1, 1, 2, 2]);
    let out = ops::conv2d(&input, &weight, (1, 1), (0, 0));
    let got = out.data();
    let expected = [-4.0f32, -4.0, -4.0, -4.0];
    for i in 0..expected.len() {
        let err = (got[i] - expected[i]).abs();
        assert!(err < 1e-6, "conv idx {} err {}", i, err);
    }
}

#[test]
fn test_square_backward_no_duplicate_propagation() {
    let x = Tensor::new(&[3.0], &[1]).set_requires_grad(true);
    let y = x.mul(&x);
    y.backward();
    let grad = x.grad().expect("x grad missing");
    let err = (grad.data()[0] - 6.0).abs();
    assert!(err < 1e-6, "square grad mismatch: {}", grad.data()[0]);
}
