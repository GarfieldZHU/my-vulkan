extern crate winit;

use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{EventLoop, ControlFlow},
    window::{Window, WindowBuilder}, 
    dpi::LogicalSize
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct HelloTriangleApplication {
    events_loop: EventLoop<()>,
    window: Arc<Window>,
}

impl HelloTriangleApplication {
    pub fn initialize() -> Self {
        let (events_loop, window) = Self::init_window();

        Self {
            events_loop,
            window,
        }
    }

    fn init_window() -> (EventLoop<()>,  Arc<Window>) {
        let events_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Aloha Vulkan")
            .with_inner_size(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
            .build(&events_loop)
            .unwrap();
        (events_loop, Arc::new(window))
    }

    fn main_loop(self) {
        self.events_loop.run( move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            println!("{:?}", event);
            
            if let Event::WindowEvent { event: WindowEvent::CloseRequested, .. } = event {
                *control_flow = ControlFlow::Exit
            }
        });
    }
}

fn main() {
    let app = HelloTriangleApplication::initialize();
    app.main_loop();
}