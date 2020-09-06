extern crate vulkano;

struct HelloTriangleApplication {

}

impl HelloTriangleApplication {
    pub fn initialize() -> Self {
        Self {

        }
    }

    fn main_loop(&mut self) {

    }
}

fn main() {
    let mut app = HelloTriangleApplication::initialize();
    app.main_loop();
}