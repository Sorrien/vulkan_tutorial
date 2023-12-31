use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
};

use shaderc;

fn main() {
   /*  let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some("main"));

    let file_paths = get_all_files("shaders");

    for path in file_paths {
        let extension = path.extension().unwrap();
        if extension != "spv" {
            let shader_kind = match extension.to_str().unwrap() {
                "frag" => shaderc::ShaderKind::Fragment,
                "vert" => shaderc::ShaderKind::Vertex,
                "comp" => shaderc::ShaderKind::Compute,
                "geom" => shaderc::ShaderKind::Geometry,
                "mesh" => shaderc::ShaderKind::Mesh,
                "rgen" => shaderc::ShaderKind::RayGeneration,
                "tesc" => shaderc::ShaderKind::TessControl,
                "tese" => shaderc::ShaderKind::TessEvaluation,
                "task" => shaderc::ShaderKind::Task,
                "rint" => shaderc::ShaderKind::Intersection,
                "rahit" => shaderc::ShaderKind::AnyHit,
                "rchit" => shaderc::ShaderKind::ClosestHit,
                "rmiss" => shaderc::ShaderKind::Miss,
                "rcall" => shaderc::ShaderKind::Callable,
                _ => shaderc::ShaderKind::InferFromSource,
            };
            let mut source_file = std::fs::File::open(path.clone()).unwrap();
            let mut source = String::new();
            println!("current path: {:?}", path);
            source_file.read_to_string(&mut source).unwrap();

            let binary_result = compiler
                .compile_into_spirv(
                    &source,
                    shader_kind,
                    path.file_name().unwrap().to_str().unwrap(),
                    "main",
                    Some(&options),
                )
                .unwrap();

            let mut new_path = String::from(path.to_str().unwrap());

            new_path.push_str(".spv");
            println!("new path: {:?}", new_path);
            let mut new_file = std::fs::File::create(new_path).unwrap();
            new_file.write_all(binary_result.as_binary_u8()).unwrap();
        }
    } */
}

fn get_all_files<P>(dir: P) -> Vec<PathBuf>
where
    P: AsRef<Path>,
{
    let mut file_paths = vec![];
    for path in std::fs::read_dir(dir).unwrap() {
        let path = path.unwrap().path();

        if path.is_file() {
            file_paths.push(path);
        } else {
            file_paths.append(&mut get_all_files(path));
        }
    }
    file_paths
}
