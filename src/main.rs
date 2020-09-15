extern crate winit;
extern crate vulkano_win;

use std::sync::Arc;
use std::collections::HashSet;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{EventLoop, ControlFlow},
    window::{Window, WindowBuilder}, 
    dpi::LogicalSize
};
use vulkano_win::VkSurfaceBuild;

use vulkano::instance::{
    Instance,
    InstanceExtensions,
    ApplicationInfo,
    Version,
    layers_list,
    debug::{DebugCallback, MessageType, MessageSeverity},
    PhysicalDevice,
};
use vulkano::device::{Device, DeviceExtensions, Queue, Features};
use vulkano::swapchain::{
    Surface,
};


const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION_LAYERS: &[&str] =  &[
    "VK_LAYER_LUNARG_standard_validation"
];

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

struct QueueFamilyIndices {
    graphics_family: i32,
    present_family: i32,
}
impl QueueFamilyIndices {
    fn new() -> Self {
        Self { graphics_family: -1, present_family: -1 }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family >= 0 && self.present_family >= 0
    }
}

#[allow(unused)]
struct HelloTriangleApplication {
    instance: Arc<Instance>,
    debug_callback: Option<DebugCallback>,
    events_loop: EventLoop<()>,
    surface: Arc<Surface<Window>>,
    window: Arc<Window>,

    physical_device_index: usize, // can't store PhysicalDevice directly (lifetime issues)
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,
}

impl HelloTriangleApplication {
    pub fn initialize() -> Self {
        let (_events_loop, window) = Self::init_window();
        let instance = Self::create_instance();
        let debug_callback = Self::setup_debug_callback(&instance);
        let (events_loop, surface) = Self::create_surface(&instance);
        
        let physical_device_index = Self::pick_physical_device(&instance, &surface);
        let (device, graphics_queue, present_queue) = Self::create_logical_device(
            &instance, &surface, physical_device_index);

        Self {
            instance,
            debug_callback,
            events_loop,
            surface,
            window,
            physical_device_index,
            device, 
            graphics_queue,
            present_queue,
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
        if ENABLE_VALIDATION_LAYERS && !Self::check_validation_layer_support() {
            println!("Validation layers requested, but not available!")
        }

        let supported_extensions = InstanceExtensions::supported_by_core()
            .expect("failed to retrieve supported extensions");
        println!("Supported extensions: {:?}", supported_extensions);

        let app_info = ApplicationInfo {
            application_name: Some("Hello Triangle".into()),
            application_version: Some(Version { major: 1, minor: 0, patch: 0 }),
            engine_name: Some("No Engine".into()),
            engine_version: Some(Version { major: 1, minor: 0, patch: 0 }),
        };

        let required_extensions = Self::get_required_extensions();
        if ENABLE_VALIDATION_LAYERS && Self::check_validation_layer_support() {
            Instance::new(Some(&app_info), &required_extensions, VALIDATION_LAYERS.iter().cloned())
                .expect("failed to create Vulkan instance")
        } else {
            Instance::new(Some(&app_info), &required_extensions, None)
                .expect("failed to create Vulkan instance")
        }
    }

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = layers_list().unwrap().map(|l| l.name().to_owned()).collect();
        VALIDATION_LAYERS.iter()
            .all(|layer_name| layers.contains(&layer_name.to_string()))
    }

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();
        if ENABLE_VALIDATION_LAYERS {
            // TODO!: this should be ext_debug_utils (_report is deprecated), but that doesn't exist yet in vulkano
            extensions.ext_debug_utils = true;
        }

        extensions
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS  {
            return None;
        }

        let msg_severity = MessageSeverity {
            error: true,
            warning: true,
            information: false,
            verbose: true,
        };

        let msg_type =  MessageType {
            general: true,
            validation: true,
            performance: true,
        };

        DebugCallback::new(&instance, msg_severity, msg_type, |msg| {
            println!("validation layer: {:?}", msg.description);
        }).ok()

    }

    fn pick_physical_device(instance: &Arc<Instance>, surface: &Arc<Surface<Window>>) -> usize {
        PhysicalDevice::enumerate(&instance)
            .position(|device| Self::is_device_suitable(surface, &device))
            .expect("failed to find a suitable GPU!")
    }

    fn is_device_suitable(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> bool {
        let indices = Self::find_queue_families(surface, device);
        indices.is_complete()
    }

    fn find_queue_families(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::new();
        // TODO: replace index with id to simplify?
        for (i, queue_family) in device.queue_families().enumerate() {
            if queue_family.supports_graphics() {
                indices.graphics_family = i as i32;
            }

            if surface.is_supported(queue_family).unwrap() {
                indices.present_family = i as i32;
            }
            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn create_logical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let indices = Self::find_queue_families(&surface, &physical_device);
        
        let families = [indices.graphics_family, indices.present_family];
        use std::iter::FromIterator;
        let unique_queue_families: HashSet<&i32> = HashSet::from_iter(families.iter());

        let queue_priority = 1.0;
        let queue_families = unique_queue_families.iter().map(|i| {
            (physical_device.queue_families().nth(**i as usize).unwrap(), queue_priority)
        });

        let (device, mut queues) = Device::new(physical_device, &Features::none(),
            &DeviceExtensions::none(), queue_families)
            .expect("failed to create logical device!");

        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

        (device, graphics_queue, present_queue)
    }

    fn create_surface(instance: &Arc<Instance>) -> (EventLoop<()>, Arc<Surface<Window>>) {
        let events_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .with_title("Vulkan")
            .with_inner_size(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
            .build_vk_surface(&events_loop, instance.clone())
            .expect("failed to create window surface!");
        (events_loop, surface)
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