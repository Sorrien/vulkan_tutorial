use std::mem;

use ash::{util::Align, vk};

#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: glam::Vec2,
    pub color: glam::Vec3,
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        //It seems like it'd be possible to automatically map most types to a requisite format and offset
        vec![
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32),
        ]
    }
}

// Simple offset_of macro akin to C++ offsetof
#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = std::mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}
pub use offset_of;

pub struct VertexBuffer {
    pub buffer: MyBuffer,
    pub vertices: Vec<Vertex>,
}

impl VertexBuffer {
    pub fn init(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        graphics_queue: vk::Queue,
        vertices: Vec<Vertex>,
    ) -> Self {
        let size = (std::mem::size_of::<Vertex>() * vertices.len()) as u64;

        let mut staging_buffer = MyBuffer::init(
            instance,
            physical_device,
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let vert_ptr = unsafe {
            device.map_memory(
                staging_buffer.memory,
                0,
                (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
                vk::MemoryMapFlags::empty(),
            )
        }
        .expect("failed to map vertex buffer!");
        let mut vert_align =
            unsafe { Align::new(vert_ptr, mem::align_of::<Vertex>() as u64, size) };
        vert_align.copy_from_slice(&vertices);
        unsafe { device.unmap_memory(staging_buffer.memory) };

        let vertex_buffer = MyBuffer::init(
            instance,
            physical_device,
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        copy_buffer(
            device,
            command_pool,
            graphics_queue,
            staging_buffer.buffer,
            vertex_buffer.buffer,
            size,
        );

        staging_buffer.cleanup(device);

        Self {
            buffer: vertex_buffer,
            vertices,
        }
    }

    pub fn cleanup(&mut self, device: &ash::Device) {
        self.buffer.cleanup(device);
    }
}

pub struct MyBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

impl MyBuffer {
    fn init(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        device: &ash::Device,
        size: vk::DeviceSize,
        buffer_usage_flags: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Self {
        let buffer = Self::create_buffer(&device, size, buffer_usage_flags, sharing_mode)
            .expect("failed to create vertex buffer!");

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let memory_type_index = Self::find_memory_type(
            instance,
            physical_device,
            mem_requirements.memory_type_bits,
            memory_properties,
        );
        let memory = Self::allocate_buffer(&device, mem_requirements, memory_type_index);

        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .expect("failed to bind vertex buffer memory!");

        Self { buffer, memory }
    }

    fn cleanup(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }

    fn create_buffer(
        device: &ash::Device,
        size: vk::DeviceSize,
        buffer_usage_flags: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
    ) -> Result<vk::Buffer, vk::Result> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(buffer_usage_flags)
            .sharing_mode(sharing_mode);

        let vertex_buffer = unsafe { device.create_buffer(&buffer_info, None) };

        vertex_buffer
    }

    fn allocate_buffer(
        device: &ash::Device,
        mem_requirements: vk::MemoryRequirements,
        memory_type_index: u32,
    ) -> vk::DeviceMemory {
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let device_memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .expect("failed to allocate vertex buffer memory!");

        device_memory
    }

    fn find_memory_type(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let mem_properties =
            unsafe { instance.get_physical_device_memory_properties(*physical_device) };
        for i in 0..(mem_properties.memory_type_count as usize) {
            if (type_filter & (1 << i)) != 0
                && (mem_properties.memory_types[i].property_flags & properties) == properties
            {
                return i as u32;
            }
        }

        panic!("failed to find suitable memory type!");
    }
}

fn copy_buffer(
    device: &ash::Device,
    command_pool: &vk::CommandPool,
    graphics_queue: vk::Queue,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    size: vk::DeviceSize,
) {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(*command_pool)
        .command_buffer_count(1);

    let command_buffers = unsafe { device.allocate_command_buffers(&allocate_info) }
        .expect("failed to create command buffer!");
    let command_buffer = command_buffers[0];
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
        .expect("failed to begin copy command buffer!");

    let copy_region = vk::BufferCopy::default()
        .src_offset(0)
        .dst_offset(0)
        .size(size);
    let regions = [copy_region];
    unsafe { device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &regions) };

    unsafe { device.end_command_buffer(command_buffer) }
        .expect("failed to end copy command buffer!");

    let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
    let submits = [submit_info];
    unsafe { device.queue_submit(graphics_queue, &submits, vk::Fence::null()) }
        .expect("failed to submit buffer copy to queue");
    unsafe { device.queue_wait_idle(graphics_queue) }
        .expect("failed to wait on queue idle after buffer copy submit!");

    unsafe { device.free_command_buffers(*command_pool, &command_buffers) };
}
