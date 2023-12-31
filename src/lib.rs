use std::{
    ffi::{c_char, c_void, CStr},
    fs::File,
    path::Path,
    ptr,
};

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    util::read_spv,
    util::*,
    vk::{
        self, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, BlendFactor, BlendOp,
        ColorComponentFlags, ColorSpaceKHR, CommandPoolCreateFlags, ComponentMapping,
        ComponentSwizzle, CompositeAlphaFlagsKHR, CullModeFlags, DeviceCreateInfo,
        DeviceQueueCreateInfo, DynamicState, Extent2D, Format, FrontFace,
        GraphicsPipelineCreateInfo, ImageAspectFlags, ImageLayout, ImageSubresourceRange,
        ImageUsageFlags, ImageView, ImageViewCreateInfo, ImageViewType, LogicOp, Offset2D,
        PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceType, Pipeline, PipelineBindPoint,
        PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
        PipelineDynamicStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineLayout,
        PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
        PresentModeKHR, PrimitiveTopology, QueueFlags, Rect2D, RenderPassCreateInfo,
        SampleCountFlags, ShaderStageFlags, SharingMode, SubpassDescription,
        SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR, SwapchainCreateInfoKHR, SwapchainKHR,
        Viewport,
    },
    Device, Entry, Instance,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

pub mod debug;

pub struct VulkanApplication {
    window: Window,
    instance: Instance,
    debug_utils: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    device: Device,
    surface: SurfaceKHR,
    surface_loader: Surface,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: SwapchainKHR,
    swapchain_loader: Swapchain,
    swapchain_images: Vec<vk::Image>,
    format: Format,
    extent: Extent2D,
    swapchain_image_views: Vec<ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: PipelineLayout,
    graphics_pipeline: Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

impl VulkanApplication {
    pub fn new(window: Window) -> Self {
        let entry = ash::Entry::linked();
        let window_size = window.inner_size();
        let instance = Self::create_instance(&window, &entry);
        let (debug_utils, debug_messenger) = crate::debug::debug_utils(&entry, &instance);

        let surface_loader = Surface::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
        }
        .expect("failed to create window surface!");

        let (physical_device, queue_family_indices, swapchain_support) =
            Self::pick_physical_device(&instance, &surface_loader, &surface)
                .expect("failed to find physical device!");

        let logical_device =
            Self::create_logical_device(&instance, &physical_device, &queue_family_indices)
                .expect("failed to create logical device!");

        let graphics_queue = unsafe {
            logical_device.get_device_queue(queue_family_indices.graphics_family.unwrap() as u32, 0)
        };

        let present_queue = unsafe {
            logical_device.get_device_queue(queue_family_indices.present_family.unwrap() as u32, 0)
        };

        let (swapchain, swapchain_loader, format, extent) = Self::create_swapchain(
            &instance,
            &logical_device,
            &surface,
            window_size.width,
            window_size.height,
            &queue_family_indices,
            &swapchain_support,
        )
        .expect("failed to create swapchain!");

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }
            .expect("failed to get swapchain images!");

        let swapchain_image_views =
            Self::create_swapchain_image_views(&logical_device, &swapchain_images, format);

        let graphics_subpass = Self::create_graphics_sub_pass();
        let render_pass =
            Self::create_render_pass(&logical_device, &format, vec![graphics_subpass])
                .expect("failed to create render pass!");

        let (pipeline_layout, graphics_pipeline) =
            Self::create_graphics_pipeline(&logical_device, &render_pass);

        let swapchain_framebuffers = Self::create_frame_buffers(
            &swapchain_image_views,
            &render_pass,
            &extent,
            &logical_device,
        );

        let command_pool = Self::create_command_pool(&logical_device, &queue_family_indices);

        let command_buffer = *Self::create_command_buffer(&logical_device, &command_pool)
            .expect("failed to allocate command buffers!")
            .first()
            .unwrap();

        let (image_available_semaphore, render_finished_semaphore, in_flight_fence) =
            Self::create_sync_objects(&logical_device);

        Self {
            window,
            instance,
            debug_utils,
            debug_messenger,
            device: logical_device,
            surface,
            surface_loader,
            graphics_queue,
            present_queue,
            swapchain,
            swapchain_loader,
            swapchain_images,
            format,
            extent,
            swapchain_image_views,
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            swapchain_framebuffers,
            command_pool,
            command_buffer,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        }
    }

    pub fn run(&mut self, event_loop: EventLoop<()>) -> Result<(), winit::error::EventLoopError> {
        self.main_loop(event_loop)
    }

    fn main_loop(&mut self, event_loop: EventLoop<()>) -> Result<(), winit::error::EventLoopError> {
        event_loop.run(move |event, elwt| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                unsafe { self.device.device_wait_idle() }
                    .expect("failed to wait for idle on exit!");
                elwt.exit()
            }
            Event::AboutToWait => {
                //AboutToWait is the new MainEventsCleared
                self.window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                window_id: _,
            } => {
                self.draw_frame();
            }
            _ => (),
        })
    }

    fn draw_frame(&self) {
        //println!("drawing new frame!");
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
        }
        .expect("failed to wait for in flight fence!");
        unsafe { self.device.reset_fences(&[self.in_flight_fence]) }
            .expect("failed to reset in flight fence!");

        let (image_index, _) = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphore,
                vk::Fence::null(),
            )
        }
        .expect("failed to acquire next swapchain image!");

        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .expect("failed to reset command buffer!");

        self.record_command_buffer(image_index as usize);
        let command_buffers = [self.command_buffer];
        let command_submit_wait_semaphores = [self.image_available_semaphore];
        let wait_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_submit_signal_semaphores = [self.render_finished_semaphore];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(&command_submit_wait_semaphores)
            .wait_dst_stage_mask(&wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(&command_submit_signal_semaphores)
            .build();

        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], self.in_flight_fence)
        }
        .expect("failed to submit draw command buffer!");

        let swapchains = [self.swapchain];
        let present_wait_semaphores = [self.render_finished_semaphore];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&present_wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .build();

        //present_info.p_results = ptr::null_mut();

        unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        }
        .expect("failed to queue present!");
    }

    fn create_instance(window: &Window, entry: &Entry) -> Instance {
        #[cfg(feature = "validation_layers")]
        let enable_validation_layers = true;
        #[cfg(not(feature = "validation_layers"))]
        let enable_validation_layers = false;

        let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"Hello Triangle\0") };
        let engine_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"No Engine\0") };
        /*
        let mut required_extension_names =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec(); */
        let mut required_extension_names = vec![];

        let mut window_extension_names =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec();

        required_extension_names.append(&mut window_extension_names);
        #[cfg(feature = "validation_layers")]
        required_extension_names.push(DebugUtils::name().as_ptr());
        #[cfg(feature = "validation_layers")]
        println!("Validation Layers enabled!");

        let extension_properties = entry
            .enumerate_instance_extension_properties(None)
            .expect("failed to enumerate instance extension props!");

        println!("Enabled extensions:");
        for extension_name in required_extension_names.iter() {
            let str = unsafe { CStr::from_ptr(*extension_name) }
                .to_str()
                .expect("failed to get ext name str");

            if extension_properties.iter().any(|prop| {
                unsafe { CStr::from_ptr(prop.extension_name.as_ptr()) }
                    .to_str()
                    .unwrap()
                    == str
            }) {
                println!("{}", str);
            } else {
                panic!("required extensions were not available!");
            }
        }

        let appinfo = vk::ApplicationInfo::builder()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let layer_names = unsafe {
            [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )]
        };

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut create_info = vk::InstanceCreateInfo::builder()
            .application_info(&appinfo)
            .enabled_extension_names(&required_extension_names);
        if enable_validation_layers {
            Self::check_validation_layer_support(&entry, &layers_names_raw);

            create_info = create_info.enabled_layer_names(&layers_names_raw);

            create_info.p_next = &debug::create_debug_info()
                as *const vk::DebugUtilsMessengerCreateInfoEXT
                as *const c_void;
        }

        let instance: Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Instance creation error")
        };

        instance
    }

    fn check_validation_layer_support(entry: &Entry, layers_names_raw: &Vec<*const c_char>) {
        let available_layers = entry
            .enumerate_instance_layer_properties()
            .expect("failed to get available layers!");

        for name in layers_names_raw.iter() {
            let str = unsafe { CStr::from_ptr(*name) }
                .to_str()
                .expect("failed to get layer name str");

            if available_layers.iter().any(|prop| {
                unsafe { CStr::from_ptr(prop.layer_name.as_ptr()) }
                    .to_str()
                    .unwrap()
                    == str
            }) {
                println!("{}", str);
            } else {
                panic!("required layers were not available!");
            }
        }
    }

    pub fn init_window(width: u32, height: u32) -> (EventLoop<()>, winit::window::Window) {
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new()
            .with_title("Vulkan Tutorial")
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .build(&event_loop)
            .unwrap();

        (event_loop, window)
    }

    fn pick_physical_device(
        instance: &Instance,
        surface_loader: &Surface,
        surface: &SurfaceKHR,
    ) -> Option<(PhysicalDevice, QueueFamilyIndices, SwapchainSupportDetails)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .expect("failed to find physical devices!");

        if physical_devices.len() == 0 {
            panic!("failed to find GPUs with Vulkan support!");
        }

        let mut scored_devices = physical_devices
            .iter()
            .filter_map(|device| {
                if let Some((score, queue_family_indices, swapchain_details)) =
                    Self::rate_device(instance, device, surface_loader, surface)
                {
                    Some((score, device, queue_family_indices, swapchain_details))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        scored_devices.sort_by(|(a, _, _, _), (b, _, _, _)| a.cmp(b));

        if let Some((_, device, queue_family_indices, swapchain_details)) = scored_devices.last() {
            Some((**device, *queue_family_indices, swapchain_details.clone()))
        } else {
            None
        }
    }

    fn rate_device(
        instance: &Instance,
        device: &PhysicalDevice,
        surface_loader: &Surface,
        surface: &SurfaceKHR,
    ) -> Option<(u32, QueueFamilyIndices, SwapchainSupportDetails)> {
        let mut score = 0;

        let features = unsafe { instance.get_physical_device_features(*device) };
        let properties = unsafe { instance.get_physical_device_properties(*device) };

        if properties.device_type == PhysicalDeviceType::DISCRETE_GPU {
            // Discrete GPUs have a significant performance advantage
            score += 1000;
        }

        // Maximum possible size of textures affects graphics quality
        score += properties.limits.max_image_dimension2_d;

        // Application can't function without geometry shaders or the graphics queue family
        /* if features.geometry_shader == 1 && queue_family_indices.graphics_family.is_some() {
            Some((score, queue_family_indices, swapchain_details))
        } else {
            None
        } */
        let device_props = unsafe { instance.get_physical_device_properties(*device) };
        let device_name =
            unsafe { CStr::from_ptr(device_props.device_name.as_ptr()).to_str() }.unwrap();
        // Application can't function without geometry shaders or the graphics queue family
        if features.geometry_shader == 1 {
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(*device) };

            let queue_family_indices =
                QueueFamilyIndices::new(&queue_families, surface_loader, device, surface);
            if queue_family_indices.graphics_family.is_some() {
                let swapchain_details =
                    SwapchainSupportDetails::new(device, surface_loader, surface);

                if swapchain_details.formats.len() > 0 && swapchain_details.present_modes.len() > 0
                {
                    Some((score, queue_family_indices, swapchain_details))
                } else {
                    println!(
                        "failed to find swapchain format or present mode on physical device! {}",
                        device_name
                    );
                    None
                }
            } else {
                println!(
                    "failed to find graphics family queue on physical device! {}",
                    device_name
                );
                None
            }
        } else {
            None
        }
    }

    fn create_logical_device(
        instance: &Instance,
        device: &PhysicalDevice,
        queue_family_indices: &QueueFamilyIndices,
    ) -> Result<ash::Device, vk::Result> {
        let queue_create_infos = [queue_family_indices.graphics_family.unwrap() as u32].map(|i| {
            DeviceQueueCreateInfo::builder()
                .queue_family_index(i)
                .queue_priorities(&[1.])
                .build()
        });

        let device_features = PhysicalDeviceFeatures::builder()
            .geometry_shader(true)
            .build();

        //may want to add check against available extensions later but the availability of present implies swap chain extension availability.
        let device_extension_names_raw = [
            Swapchain::name().as_ptr(),
            //#[cfg(any(target_os = "macos", target_os = "ios"))]
            //KhrPortabilitySubsetFn::NAME.as_ptr(),
        ];
        let device_create_info = DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extension_names_raw)
            .build(); //device specific layers are outdated but keep in mind if an issue crops up on older hardware

        unsafe { instance.create_device(*device, &device_create_info, None) }
    }

    fn create_swapchain(
        instance: &Instance,
        logical_device: &Device,
        surface: &SurfaceKHR,
        window_width: u32,
        window_height: u32,
        queue_family_indices: &QueueFamilyIndices,
        swapchain_support: &SwapchainSupportDetails,
    ) -> Result<(vk::SwapchainKHR, Swapchain, Format, Extent2D), vk::Result> {
        let surface_format = Self::choose_swapchain_surface_format(&swapchain_support.formats)
            .expect("failed to find surface format!");
        let present_mode = Self::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent =
            Self::choose_swap_extent(&swapchain_support.capabilities, window_width, window_height);

        let image_count = swapchain_support.capabilities.min_image_count + 1;

        //if max_image_count is 0 then there is no max
        let image_count = if swapchain_support.capabilities.max_image_count > 0
            && image_count > swapchain_support.capabilities.max_image_count
        {
            swapchain_support.capabilities.max_image_count
        } else {
            image_count
        };

        let (sharing_mode, queue_indices) =
            if queue_family_indices.graphics_family != queue_family_indices.present_family {
                (
                    SharingMode::CONCURRENT,
                    vec![
                        queue_family_indices.graphics_family.unwrap() as u32,
                        queue_family_indices.present_family.unwrap() as u32,
                    ],
                )
            } else {
                (SharingMode::EXCLUSIVE, vec![])
            };

        let swapchain_loader = Swapchain::new(instance, logical_device);
        let swapchain_create_info = SwapchainCreateInfoKHR::builder()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(&queue_indices)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }?;

        Ok((swapchain, swapchain_loader, surface_format.format, extent))
    }

    fn choose_swapchain_surface_format(
        available_formats: &Vec<SurfaceFormatKHR>,
    ) -> Option<&SurfaceFormatKHR> {
        if let Some(desired_format) = available_formats.iter().find(|format| {
            format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
                && format.format == Format::B8G8R8A8_SRGB
        }) {
            Some(desired_format)
        } else {
            available_formats.first()
        }
    }

    fn choose_swapchain_present_mode(
        available_present_modes: &Vec<PresentModeKHR>,
    ) -> PresentModeKHR {
        let desired_mode = PresentModeKHR::MAILBOX;
        let is_desired_mode_available = available_present_modes
            .iter()
            .any(|present_mode| *present_mode == desired_mode);
        if is_desired_mode_available {
            desired_mode
        } else {
            PresentModeKHR::FIFO
        }
    }

    fn choose_swap_extent(
        capabilities: &SurfaceCapabilitiesKHR,
        window_width: u32,
        window_height: u32,
    ) -> Extent2D {
        match capabilities.current_extent.width {
            //the max value of u32 is a special value to indicate that we must choose a resolution with the current min and max extents
            //should look into how DPI scaling is handled by winit and if this is the pixel extent or if this includes dpi scaling.
            u32::MAX => vk::Extent2D {
                width: window_width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: window_height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            },
            _ => capabilities.current_extent,
        }
    }

    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &Vec<vk::Image>,
        swapchain_image_format: Format,
    ) -> Vec<ImageView> {
        let component_mapping = ComponentMapping::builder()
            .r(ComponentSwizzle::IDENTITY)
            .g(ComponentSwizzle::IDENTITY)
            .b(ComponentSwizzle::IDENTITY)
            .a(ComponentSwizzle::IDENTITY);
        let subresource_range = ImageSubresourceRange::builder()
            .aspect_mask(ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        swapchain_images
            .iter()
            .map(|image| {
                *ImageViewCreateInfo::builder()
                    .image(*image)
                    .view_type(ImageViewType::TYPE_2D)
                    .format(swapchain_image_format)
                    .components(*component_mapping)
                    .subresource_range(*subresource_range)
            })
            .map(|image_view_create_info| {
                unsafe { device.create_image_view(&image_view_create_info, None) }
                    .expect("failed to create image view for swapchain image!")
            })
            .collect::<Vec<_>>()
    }

    fn create_shader_module<P>(path: P, device: &Device) -> vk::ShaderModule
    where
        P: AsRef<Path>,
    {
        let mut spv_file = File::open(path).unwrap();
        let shader_code = read_spv(&mut spv_file).expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::builder().code(&shader_code);
        let shader_module = unsafe {
            device
                .create_shader_module(&vertex_shader_info, None)
                .expect("Vertex shader module error")
        };
        shader_module
    }

    fn create_graphics_pipeline(
        device: &Device,
        render_pass: &vk::RenderPass,
    ) -> (PipelineLayout, Pipeline) {
        let vertex_shader_module =
            Self::create_shader_module("shaders/triangle/test.vert.spv", device);
        let fragment_shader_module =
            Self::create_shader_module("shaders/triangle/test.frag.spv", device);
        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let vertex_shader_stage_info = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(shader_entry_name)
            .build();

        let fragment_shader_stage_info = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(shader_entry_name)
            .build();

        let shader_stages = vec![vertex_shader_stage_info, fragment_shader_stage_info];

        let dynamic_state = PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .build();

        let vertex_input_info = PipelineVertexInputStateCreateInfo::builder().build();

        let input_assembly = PipelineInputAssemblyStateCreateInfo::builder()
            .topology(PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1)
            .build();

        let rasterizer = PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(PolygonMode::FILL)
            .line_width(1.)
            .cull_mode(CullModeFlags::BACK)
            .front_face(FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .build();

        let multisampling = PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(SampleCountFlags::TYPE_1)
            .build();

        let color_blend_attachment = PipelineColorBlendAttachmentState::builder()
            .color_write_mask(ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD)
            .build();

        let color_blending = PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(&[color_blend_attachment])
            .build();

        let pipeline_layout_info = PipelineLayoutCreateInfo::builder().build();

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }.unwrap();

        let pipeline_info = GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            //.depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0)
            .build();

        let graphics_pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .expect("failed to create graphics pipeline!");

        unsafe {
            device.destroy_shader_module(fragment_shader_module, None);
            device.destroy_shader_module(vertex_shader_module, None);
        }

        (pipeline_layout, *graphics_pipelines.first().unwrap())
    }

    fn create_graphics_sub_pass() -> vk::SubpassDescription {
        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let mut subpass = vk::SubpassDescription::default()
            .color_attachments(&color_attachment_refs)
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .resolve_attachments(&[]);
        subpass
    }

    fn create_render_pass(
        device: &Device,
        swapchain_image_format: &Format,
        subpasses: Vec<SubpassDescription>,
    ) -> Result<vk::RenderPass, vk::Result> {
        let color_attachment = AttachmentDescription::default()
            .format(*swapchain_image_format)
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .build();

        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];

        let color_attachments = [color_attachment];

        let render_pass_info = RenderPassCreateInfo::builder()
            .attachments(&color_attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies)
            .build();

        unsafe { device.create_render_pass(&render_pass_info, None) }
    }

    fn create_frame_buffers(
        swapchain_image_views: &Vec<ImageView>,
        render_pass: &vk::RenderPass,
        swapchain_extent: &Extent2D,
        device: &Device,
    ) -> Vec<vk::Framebuffer> {
        swapchain_image_views
            .iter()
            .map(|swapchain_image_view| {
                let attachments = [*swapchain_image_view];
                vk::FramebufferCreateInfo::builder()
                    .render_pass(*render_pass)
                    .attachments(&attachments)
                    .width(swapchain_extent.width)
                    .height(swapchain_extent.height)
                    .layers(1)
                    .build()
            })
            .map(|framebuffer_info| {
                unsafe { device.create_framebuffer(&framebuffer_info, None) }
                    .expect("failed to create framebuffer!")
            })
            .collect::<Vec<_>>()
    }

    fn create_command_pool(
        device: &Device,
        queue_family_indices: &QueueFamilyIndices,
    ) -> vk::CommandPool {
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_indices.graphics_family.unwrap() as u32)
            .build();

        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .expect("failed to create command pool!");
        command_pool
    }

    fn create_command_buffer(
        device: &Device,
        command_pool: &vk::CommandPool,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
            .build();

        unsafe { device.allocate_command_buffers(&alloc_info) }
    }

    fn record_command_buffer(&self, image_index: usize) {
        let begin_info = vk::CommandBufferBeginInfo::builder().build();

        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
        }
        .expect("failed to begin recording command buffer!");

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        }];

        let render_pass_begin = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain_framebuffers[image_index])
            .render_area(Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: self.extent,
            })
            .clear_values(&clear_values)
            .build();

        unsafe {
            self.device.cmd_begin_render_pass(
                self.command_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            )
        };

        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            )
        };

        let viewport = Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(self.extent.width as f32)
            .height(self.extent.height as f32)
            .min_depth(0.)
            .max_depth(1.)
            .build();
        unsafe {
            self.device
                .cmd_set_viewport(self.command_buffer, 0, &[viewport])
        };

        let scissor = Rect2D::builder()
            .offset(Offset2D { x: 0, y: 0 })
            .extent(self.extent)
            .build();
        unsafe {
            self.device
                .cmd_set_scissor(self.command_buffer, 0, &[scissor])
        };

        unsafe { self.device.cmd_draw(self.command_buffer, 3, 1, 0, 0) };

        unsafe { self.device.cmd_end_render_pass(self.command_buffer) };

        unsafe { self.device.end_command_buffer(self.command_buffer) }
            .expect("failed to record command buffer!");
    }

    fn create_sync_objects(device: &Device) -> (vk::Semaphore, vk::Semaphore, vk::Fence) {
        let semaphore_create_info = vk::SemaphoreCreateInfo::builder().build();
        let image_available_semaphore =
            unsafe { device.create_semaphore(&semaphore_create_info, None) }
                .expect("failed to create semaphore!");
        let render_finished_semaphore =
            unsafe { device.create_semaphore(&semaphore_create_info, None) }
                .expect("failed to create semaphore!");

        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();
        let in_flight_fence = unsafe { device.create_fence(&fence_create_info, None) }
            .expect("failed to create fence!");

        (
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        )
    }
}

impl Drop for VulkanApplication {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.device.destroy_fence(self.in_flight_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain_framebuffers
                .iter()
                .for_each(|framebuffer| self.device.destroy_framebuffer(*framebuffer, None));
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain_image_views
                .iter()
                .for_each(|image_view| self.device.destroy_image_view(*image_view, None));
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            #[cfg(feature = "validation_layers")]
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone, Copy)]
struct QueueFamilyIndices {
    graphics_family: Option<usize>,
    present_family: Option<usize>,
}

impl QueueFamilyIndices {
    fn new(
        queue_families: &Vec<vk::QueueFamilyProperties>,
        surface_loader: &Surface,
        pdevice: &PhysicalDevice,
        surface: &SurfaceKHR,
    ) -> Self {
        let graphics_family_index = if let Some((graphics_family_index, _)) = queue_families
            .iter()
            .enumerate()
            .find(|(_, queue)| queue.queue_flags.contains(QueueFlags::GRAPHICS))
        {
            Some(graphics_family_index)
        } else {
            None
        };

        let present_family_index = if let Some((present_family_index, _)) =
            queue_families.iter().enumerate().find(|(i, _)| {
                unsafe {
                    surface_loader
                        .get_physical_device_surface_support(*pdevice, *i as u32, *surface)
                }
                .unwrap()
            }) {
            Some(present_family_index)
        } else {
            None
        };

        Self {
            graphics_family: graphics_family_index,
            present_family: present_family_index,
        }
    }
}

#[derive(Clone)]
struct SwapchainSupportDetails {
    capabilities: SurfaceCapabilitiesKHR,
    formats: Vec<SurfaceFormatKHR>,
    present_modes: Vec<PresentModeKHR>,
}

impl SwapchainSupportDetails {
    fn new(device: &PhysicalDevice, surface_loader: &Surface, surface: &SurfaceKHR) -> Self {
        let capabilities =
            unsafe { surface_loader.get_physical_device_surface_capabilities(*device, *surface) }
                .expect("failed to get surface capabilites!");
        let formats =
            unsafe { surface_loader.get_physical_device_surface_formats(*device, *surface) }
                .expect("failed to get device surface formats!");
        let present_modes =
            unsafe { surface_loader.get_physical_device_surface_present_modes(*device, *surface) }
                .expect("failed to get device surface present modes!");

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }
}
