use std::{
    ffi::{c_char, c_void, CStr},
    fs::File,
    path::Path,
};

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    util::read_spv,
    vk::{
        self, AttachmentDescription, AttachmentLoadOp, AttachmentStoreOp, BlendFactor, BlendOp,
        ColorComponentFlags, ColorSpaceKHR, CompositeAlphaFlagsKHR, CullModeFlags,
        DeviceCreateInfo, DeviceQueueCreateInfo, DynamicState, Extent2D, Format, FrontFace,
        GraphicsPipelineCreateInfo, ImageLayout, ImageUsageFlags, ImageView, LogicOp,
        PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceType,
        PipelineColorBlendAttachmentState, PipelineColorBlendStateCreateInfo,
        PipelineDynamicStateCreateInfo, PipelineInputAssemblyStateCreateInfo,
        PipelineLayoutCreateInfo, PipelineMultisampleStateCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
        PresentModeKHR, PrimitiveTopology, RenderPassCreateInfo, SampleCountFlags,
        ShaderStageFlags, SharingMode, SubpassDescription, SurfaceCapabilitiesKHR,
        SurfaceFormatKHR, SurfaceKHR, SwapchainCreateInfoKHR,
    },
    Entry, Instance,
};
use winit::window::Window;

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use crate::{buffers::Vertex, debug, VulkanApplication};

pub struct BaseVulkanState {
    pub window: Window,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue_family_indices: QueueFamilyIndices,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,
    pub swapchain_support: SwapchainSupportDetails,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub debug_utils: DebugUtils,
}

impl BaseVulkanState {
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

        let device =
            Self::create_logical_device(&instance, &physical_device, &queue_family_indices)
                .expect("failed to create logical device!");

        let graphics_queue = unsafe {
            device.get_device_queue(queue_family_indices.graphics_family.unwrap() as u32, 0)
        };

        let present_queue = unsafe {
            device.get_device_queue(queue_family_indices.present_family.unwrap() as u32, 0)
        };

        Self {
            window,
            instance,
            physical_device,
            device,
            queue_family_indices,
            graphics_queue,
            present_queue,
            swapchain_support,
            surface,
            surface_loader,
            debug_utils,
            debug_messenger,
        }
    }

    fn create_instance(window: &Window, entry: &Entry) -> Instance {
        #[cfg(feature = "validation_layers")]
        let enable_validation_layers = true;
        #[cfg(not(feature = "validation_layers"))]
        let enable_validation_layers = false;

        let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"Hello Triangle\0") };
        let engine_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"No Engine\0") };

        let mut required_extension_names = vec![];

        let mut window_extension_names =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec();

        required_extension_names.append(&mut window_extension_names);
        #[cfg(feature = "validation_layers")]
        required_extension_names.push(DebugUtils::NAME.as_ptr());
        #[cfg(feature = "validation_layers")]
        println!("Validation Layers enabled!");

        let extension_properties = unsafe { entry.enumerate_instance_extension_properties(None) }
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
        println!("");

        let appinfo = vk::ApplicationInfo::default()
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

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_extension_names(&required_extension_names);

        let debug_info = debug::create_debug_info();
        if enable_validation_layers {
            Self::check_validation_layer_support(&entry, &layers_names_raw);
            create_info = create_info.enabled_layer_names(&layers_names_raw);

            create_info.p_next =
                &debug_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;
        }

        let instance: Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Instance creation error")
        };

        instance
    }

    fn check_validation_layer_support(entry: &Entry, layers_names_raw: &Vec<*const c_char>) {
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
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

        let device_props = unsafe { instance.get_physical_device_properties(*device) };
        let device_name =
            unsafe { CStr::from_ptr(device_props.device_name.as_ptr()).to_str() }.unwrap();
        // Application can't function without geometry shaders or the graphics queue family or anisotropy (we could remove anisotropy)
        if features.geometry_shader == 1 && features.sampler_anisotropy == 1 {
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
            DeviceQueueCreateInfo::default()
                .queue_family_index(i)
                .queue_priorities(&[1.])
        });

        let device_features = PhysicalDeviceFeatures::default()
            .geometry_shader(true)
            .sampler_anisotropy(true);

        //may want to add check against available extensions later but the availability of present implies swap chain extension availability.
        let device_extension_names_raw = [
            Swapchain::NAME.as_ptr(),
            //#[cfg(any(target_os = "macos", target_os = "ios"))]
            //KhrPortabilitySubsetFn::NAME.as_ptr(),
        ];
        let device_create_info = DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extension_names_raw); //device specific layers are outdated but keep in mind if an issue crops up on older hardware

        unsafe { instance.create_device(*device, &device_create_info, None) }
    }

    pub fn create_swapchain(
        &self,
        window_width: u32,
        window_height: u32,
    ) -> Result<(vk::SwapchainKHR, Swapchain, Format, Extent2D), vk::Result> {
        let surface_format = Self::choose_swapchain_surface_format(&self.swapchain_support.formats)
            .expect("failed to find surface format!");
        let present_mode =
            Self::choose_swapchain_present_mode(&self.swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(
            &self.swapchain_support.capabilities,
            window_width,
            window_height,
        );

        let image_count = self.swapchain_support.capabilities.min_image_count + 1;

        //if max_image_count is 0 then there is no max
        let image_count = if self.swapchain_support.capabilities.max_image_count > 0
            && image_count > self.swapchain_support.capabilities.max_image_count
        {
            self.swapchain_support.capabilities.max_image_count
        } else {
            image_count
        };

        let (sharing_mode, queue_indices) = if self.queue_family_indices.graphics_family
            != self.queue_family_indices.present_family
        {
            (
                SharingMode::CONCURRENT,
                vec![
                    self.queue_family_indices.graphics_family.unwrap() as u32,
                    self.queue_family_indices.present_family.unwrap() as u32,
                ],
            )
        } else {
            (SharingMode::EXCLUSIVE, vec![])
        };

        let swapchain_loader = Swapchain::new(&self.instance, &self.device);
        let swapchain_create_info = SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(sharing_mode)
            .queue_family_indices(&queue_indices)
            .pre_transform(self.swapchain_support.capabilities.current_transform)
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

    pub fn create_swapchain_image_views(
        &self,
        swapchain_images: &Vec<vk::Image>,
        swapchain_image_format: Format,
    ) -> Vec<ImageView> {
        swapchain_images
            .iter()
            .map(|image| {
                VulkanApplication::create_image_view(&self.device, *image, swapchain_image_format)
            })
            .collect::<Vec<_>>()
    }

    pub fn create_render_pass(
        &self,
        subpasses: Vec<SubpassDescription>,
        format: vk::Format,
    ) -> Result<vk::RenderPass, vk::Result> {
        let color_attachment = AttachmentDescription::default()
            .format(format)
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::CLEAR)
            .store_op(AttachmentStoreOp::STORE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR);

        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];

        let color_attachments = [color_attachment];

        let render_pass_info = RenderPassCreateInfo::default()
            .attachments(&color_attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        unsafe { self.device.create_render_pass(&render_pass_info, None) }
    }

    pub fn create_descriptor_set_layout(&self) -> Vec<vk::DescriptorSetLayout> {
        let descriptor_set_layout = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = [descriptor_set_layout, sampler_layout_binding];

        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
        }
        .expect("failed to create descriptor set layout!");

        let descriptor_set_layouts = vec![descriptor_set_layout];
        descriptor_set_layouts
    }

    pub fn create_graphics_pipeline(
        &self,
        render_pass: &vk::RenderPass,
        descriptor_set_layouts: &Vec<vk::DescriptorSetLayout>,
    ) -> (vk::PipelineLayout, vk::Pipeline) {
        let vertex_shader_module = self.create_shader_module("shaders/triangle/triangle.vert.spv");
        let fragment_shader_module =
            self.create_shader_module("shaders/triangle/triangle.frag.spv");
        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

        let vertex_shader_stage_info = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(shader_entry_name);

        let fragment_shader_stage_info = PipelineShaderStageCreateInfo::default()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(shader_entry_name);

        let shader_stages = vec![vertex_shader_stage_info, fragment_shader_stage_info];

        let dynamic_state = PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR]);

        let vertex_attrib_descs = Vertex::get_attribute_descriptions();
        let vertex_bind_desc = Vertex::get_binding_description();
        let vertex_bind_descs = [vertex_bind_desc];

        let vertex_input_info = PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_bind_descs)
            .vertex_attribute_descriptions(&vertex_attrib_descs);

        let input_assembly = PipelineInputAssemblyStateCreateInfo::default()
            .topology(PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(PolygonMode::FILL)
            .line_width(1.)
            .cull_mode(CullModeFlags::BACK)
            .front_face(FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(SampleCountFlags::TYPE_1);

        let color_blend_attachment = PipelineColorBlendAttachmentState::default()
            .color_write_mask(ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD);

        let color_blend_attachments = [color_blend_attachment];
        let color_blending = PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(&color_blend_attachments);

        let pipeline_layout_info =
            PipelineLayoutCreateInfo::default().set_layouts(descriptor_set_layouts);

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .unwrap();

        let pipeline_info = GraphicsPipelineCreateInfo::default()
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
            .subpass(0);

        let graphics_pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .expect("failed to create graphics pipeline!");

        unsafe {
            self.device
                .destroy_shader_module(fragment_shader_module, None);
            self.device
                .destroy_shader_module(vertex_shader_module, None);
        }

        (pipeline_layout, *graphics_pipelines.first().unwrap())
    }

    fn create_shader_module<P>(&self, path: P) -> vk::ShaderModule
    where
        P: AsRef<Path>,
    {
        let mut spv_file = File::open(path).unwrap();
        let shader_code = read_spv(&mut spv_file).expect("Failed to read vertex shader spv file");
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&vertex_shader_info, None)
                .expect("Vertex shader module error")
        };
        shader_module
    }
}

impl Drop for BaseVulkanState {
    fn drop(&mut self) {
        unsafe {
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
pub struct QueueFamilyIndices {
    pub graphics_family: Option<usize>,
    pub present_family: Option<usize>,
}

impl QueueFamilyIndices {
    fn new(
        queue_families: &Vec<vk::QueueFamilyProperties>,
        surface_loader: &Surface,
        pdevice: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
    ) -> Self {
        let graphics_family_index = if let Some((graphics_family_index, _)) = queue_families
            .iter()
            .enumerate()
            .find(|(_, queue)| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))
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
pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        device: &vk::PhysicalDevice,
        surface_loader: &Surface,
        surface: &vk::SurfaceKHR,
    ) -> Self {
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
