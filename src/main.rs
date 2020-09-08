extern crate winit;
extern crate vulkano_win;

use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{EventLoop, ControlFlow},
    window::{Window, WindowBuilder}, 
    dpi::LogicalSize
};

use vulkano::instance::{
    Instance,
    InstanceExtensions,
    ApplicationInfo,
    Version,
};


const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

#[allow(unused)]
struct HelloTriangleApplication {
    instance: Arc<Instance>,
    events_loop: EventLoop<()>,
    window: Arc<Window>,
}

impl HelloTriangleApplication {
    pub fn initialize() -> Self {
        let (events_loop, window) = Self::init_window();
        let instance = Self::create_instance();

        Self {
            instance,
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

    fn create_instance() -> Arc<Instance> {
        let supported_extensions = InstanceExtensions::supported_by_core()
            .expect("failed to retrieve supported extensions");
        println!("Supported extensions: {:?}", supported_extensions);

        let app_info = ApplicationInfo {
            application_name: Some("Hello Triangle".into()),
            application_version: Some(Version { major: 1, minor: 0, patch: 0 }),
            engine_name: Some("No Engine".into()),
            engine_version: Some(Version { major: 1, minor: 0, patch: 0 }),
        };

        let required_extensions = vulkano_win::required_extensions();
        Instance::new(Some(&app_info), &required_extensions, None)
            .expect("failed to create Vulkan instance")
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