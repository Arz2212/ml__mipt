use nalgebra::DMatrix;
use rand::thread_rng;
use rand_distr::{Uniform, Normal, Distribution};
use plotters::prelude::*;
use rand::seq::SliceRandom;

#[derive(Clone, Copy)]
struct D3point {
    x: f64,
    y: f64,
    z: f64
}
fn gen_semp(n: usize) -> Vec<D3point>{
    let mut rng = thread_rng();
    let mean = 1.0;
    let std_dev = 3.0;
    let normal = Normal::new(mean, std_dev).unwrap();
    let uniform = Uniform::new(-0.1, 0.1);
    let mut v = Vec::new();
    for _ in 0..n{
        let x: f64 = normal.sample(&mut rng);
        let y: f64 = normal.sample(&mut rng);
        let z: f64 = 0.1 * y - 2.0 * x + uniform.sample(&mut rng);
        let g = D3point{x, y, z};
        v.push(g);
    }
    v
}

 
fn rid_reg(v: &[D3point], n: usize) -> DMatrix<f64>{
    let mut x_mat = DMatrix::zeros(n, 3);
    let alf = 0.1;
    let mut z_mat = DMatrix::zeros(n, 1);
    let i_mat = DMatrix::identity(3, 3);
    for (i, point) in v.iter().enumerate(){
        x_mat[(i, 0)] = 1.0;
        x_mat[(i, 1)] = point.x;
        x_mat[(i, 2)] = point.y;
        z_mat[(i, 0)] = point.z;
    }
    let x_mat_trans = x_mat.transpose();
    let x_mult = &x_mat_trans * &x_mat;
    let x_z_mult = &x_mat_trans * &z_mat;
    let xtx_reg = x_mult + &i_mat * alf;
    let xtx_reg =  xtx_reg.try_inverse().unwrap();
    &xtx_reg * &x_z_mult

}

fn ransac_rid_reg(v: &[D3point], _n: usize, z: f64) -> DMatrix<f64> {
    let mut rng = thread_rng();
    let mut v_mat: Vec<D3point> = Vec::new();
    let amount = 3;
    let mut best = 0;
    let mut count = 0;
    let mut best_pl: Vec<D3point> = Vec::new();
    for _ in 0..1000{
        let sample: Vec<D3point> = v.choose_multiple(&mut rng, amount).copied().collect();
        let test_pl: DMatrix<f64> = rid_reg(&sample, amount);
        for (_i, point) in v.iter().enumerate(){
            let z_pred: f64 = test_pl[(0,0)] + test_pl[(1,0)] * point.x + test_pl[(2,0)] * point.y;
            let z_real: f64 = point.z;
            if (z_pred - z_real).abs() <= z{
                count += 1;
                v_mat.push(point.clone());
            }
        }
        if count > best{
            best = count;
            best_pl = v_mat.clone();
        }
        v_mat.clear();
        count = 0;

    }

    rid_reg(&best_pl, best)
    
}
fn mse_test(v: &[D3point], n: usize, vesa: &DMatrix<f64>) -> f64{
    let mut summ: f64 = 0.0;
    for (_i, point) in v.iter().enumerate(){
        let z_pred: f64 = vesa[(0,0)] + vesa[(1,0)] * point.x + vesa[(2,0)] * point.y;
        let z_real = point.z;
        summ += (z_real - z_pred) * (z_real - z_pred);
    }
    summ/(n as f64 )
    

}

fn plot_results(v: &[D3point], bet: &DMatrix<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("graf.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    // Создаём 3D холст: поворот по оси X = 30°, по оси Z = 45°
    let mut chart = ChartBuilder::on(&root)
        .caption("Ridge Regression 3D Fit", ("sans-serif", 30))
        .build_cartesian_3d(-8.0..10.0, -25.0..20.0, -8.0..10.0)?; // X: [-8, 10], Z: [-25, 20], Y: [-8, 10]


    chart.configure_axes().draw()?;

    // 1. Отрисовка исходных точек 3D (синие точки)
    // В Plotters координаты передаются в порядке (X, Z, Y) для правильной проекции осей
    chart.draw_series(
        v.iter().map(|p| Circle::new((p.x, p.z, p.y), 2, BLUE.filled()))
    )?;

    // 2. Отрисовка регрессионной плоскости z = b0 + b1*x + b2*y
    let b0 = bet[(0, 0)];
    let b1 = bet[(1, 0)];
    let b2 = bet[(2, 0)];

    // Построение сетки по X и Y от -8 до 8 с шагом 1.0
    let step = 1.0;
    let range = -8..8;

    for x_i in range.clone() {
        for y_i in range.clone() {
            let x1 = x_i as f64;
            let y1 = y_i as f64;
            let x2 = x1 + step;
            let y2 = y1 + step;

            // Вычисляем Z для четырёх углов ячейки сетки
            let z11 = b0 + b1 * x1 + b2 * y1;
            let z12 = b0 + b1 * x1 + b2 * y2;
            let z21 = b0 + b1 * x2 + b2 * y1;
            let z22 = b0 + b1 * x2 + b2 * y2;

            // Рисуем границы ячейки (красная проволочная сетка)
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x1, z11, y1), (x1, z12, y2), (x2, z22, y2), (x2, z21, y1), (x1, z11, y1)],
                &RED,
            )))?;
        }
    }

    root.present()?;
    println!("График успешно сохранен в файл: graf.png" );
    Ok(())
}

fn main() {
    let semp: Vec<D3point> = gen_semp(100000);
     let bet1: DMatrix<f64> = rid_reg(&semp, 100000);
    let bet: DMatrix<f64> = ransac_rid_reg(&semp, 100000, 0.1);
    let  _f = plot_results(&semp, &bet);
    let semp2: Vec<D3point> = gen_semp(100000);
    println!("{}", mse_test(&semp, 100000, &bet));
    println!("{}", mse_test(&semp, 100000, &bet1));

    println!("{}", mse_test(&semp2, 100000, &bet));
    println!("{}", mse_test(&semp2, 100000, &bet1));


}