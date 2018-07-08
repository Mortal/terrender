use std::fmt::Debug;
use std::ops::{Add, Sub, Mul, Div};

#[derive(Debug, Copy, Clone)]
pub struct Point2 {
    x: f64,
    y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self {
        Point2 { x: x, y: y }
    }
}

impl Add<Point2> for Point2 {
    type Output = Point2;

    fn add(self, other: Point2) -> Point2 {
        Point2::new(self.x + other.x, self.y + other.y)
    }
}

impl Sub<Point2> for Point2 {
    type Output = Point2;

    fn sub(self, other: Point2) -> Point2 {
        Point2::new(self.x - other.x, self.y - other.y)
    }
}

impl Mul<Point2> for f64 {
    type Output = Point2;

    fn mul(self, other: Point2) -> Point2 {
        Point2::new(self * other.x, self * other.y)
    }
}

pub trait Vertex: Debug + Clone
+ Add<Self, Output=Self>
+ Sub<Self, Output=Self>
{
    fn min(&self, other: &Self) -> Self;
    fn max(&self, other: &Self) -> Self;

    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn xy(&self) -> Point2;
    fn with_point(&self, xy: Point2) -> Self;
    fn with_xy(&self, x: f64, y: f64) -> Self;

    fn scale(&self, a: f64) -> Self;
}

impl Vertex for Point2 {
    fn min(&self, other: &Self) -> Self { Point2::new(self.x.min(other.x), self.y.min(other.y)) }
    fn max(&self, other: &Self) -> Self { Point2::new(self.x.max(other.x), self.y.max(other.y)) }

    fn x(&self) -> f64 { self.x }
    fn y(&self) -> f64 { self.y }
    fn xy(&self) -> Point2 { self.clone() }
    fn with_point(&self, xy: Point2) -> Self { xy }
    fn with_xy(&self, x: f64, y: f64) -> Self { Point2::new(x, y) }

    fn scale(&self, a: f64) -> Self { Point2::new(a * self.x, a * self.y) }
}

#[derive(Debug, Copy, Clone)]
pub struct Vertex3 {
    x: f64,
    y: f64,
    z: f64,
}

fn finite_min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

fn finite_max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

impl Vertex3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        assert!(x.is_finite());
        assert!(y.is_finite());
        assert!(z.is_finite());
        Self { x: x, y: y, z: z }
    }

    pub fn get(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }

    pub fn z(&self) -> f64 { self.z }
}

impl Add<Vertex3> for Vertex3 {
    type Output = Vertex3;

    fn add(self, other: Vertex3) -> Vertex3 {
        Vertex3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub<Vertex3> for Vertex3 {
    type Output = Vertex3;

    fn sub(self, other: Vertex3) -> Vertex3 {
        Vertex3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Div<f64> for Vertex3 {
    type Output = Vertex3;

    fn div(self, denom: f64) -> Vertex3 {
        Vertex3::new(self.x / denom, self.y / denom, self.z / denom)
    }
}

impl Vertex for Vertex3 {
    fn min(&self, other: &Vertex3) -> Vertex3 {
        Vertex3 {
            x: finite_min(self.x, other.x),
            y: finite_min(self.y, other.y),
            z: finite_min(self.z, other.z),
        }
    }

    fn max(&self, other: &Vertex3) -> Vertex3 {
        Vertex3 {
            x: finite_max(self.x, other.x),
            y: finite_max(self.y, other.y),
            z: finite_max(self.z, other.z),
        }
    }

    fn x(&self) -> f64 { self.x }
    fn y(&self) -> f64 { self.y }
    fn xy(&self) -> Point2 { Point2::new(self.x, self.y) }
    fn with_point(&self, xy: Point2) -> Vertex3 {
        Vertex3::new(xy.x, xy.y, self.z)
    }
    fn with_xy(&self, x: f64, y: f64) -> Vertex3 {
        Vertex3::new(x, y, self.z)
    }

    fn scale(&self, a: f64) -> Self { Vertex3::new(a * self.x, a * self.y, a * self.z) }
}

#[derive(Debug, Clone)]
pub struct Rectangle<V: Vertex> {
    lo: V,
    hi: V,
}

#[derive(Debug, Clone)]
pub struct Face<V: Vertex> {
    a: V,
    b: V,
    c: V,
}

impl<V: Vertex> Face<V> {
    pub fn new(a: V, b: V, c: V) -> Self {
        Self { a: a, b: b, c: c }
    }

    pub fn bbox(&self) -> Rectangle<V> {
        Rectangle {
            lo: self.a.min(&self.b).min(&self.c),
            hi: self.a.max(&self.b).max(&self.c),
        }
    }

    pub fn a(&self) -> &V { &self.a }
    pub fn b(&self) -> &V { &self.b }
    pub fn c(&self) -> &V { &self.c }

    pub fn bca(&self) -> Self { Self::new(self.b.clone(), self.c.clone(), self.a.clone()) }
    pub fn cab(&self) -> Self { self.bca().bca() }
}

impl<V: Vertex> Face<V> where V: Div<f64, Output=V> {

    pub fn midpoint(&self) -> V {
        (self.a.clone() + self.b.clone() + self.c.clone()) / 3.0
    }
}

pub type Face3 = Face<Vertex3>;
pub type Rectangle3 = Rectangle<Vertex3>;

struct Interval(f64, f64);

impl Interval {
    fn disjoint(&self, other: &Interval) -> bool {
        self.1 < other.0 || other.1 < self.0
    }
}

impl Rectangle3 {
    fn xs(&self) -> Interval {
        Interval(self.lo.x, self.hi.x)
    }

    fn ys(&self) -> Interval {
        Interval(self.lo.y, self.hi.y)
    }

    pub fn disjoint_xy(&self, other: &Rectangle3) -> bool {
        self.xs().disjoint(&other.xs()) || self.ys().disjoint(&other.ys())
    }

    pub fn behind(&self, other: &Rectangle3) -> bool {
        self.hi.z < other.lo.z
    }
}
