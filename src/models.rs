use std::{collections::HashMap, path::Path};

use glam::{Vec2, Vec3};
use crate::buffers::Vertex;

pub fn load_model<P>(path: P) -> (Vec<Vertex>, Vec<u32>)
where
    P: AsRef<Path> + std::fmt::Debug,
{
    let obj = obj::Obj::load(path).expect("failed to load model!");

    let mut vertices = vec![];
    let mut indices = vec![];

    let mut vertex_hashmap: HashMap<(usize, usize), u32> = HashMap::new();

    let deduplicate = true;

    for object in obj.data.objects {
        for group in object.groups {
            for poly in group.polys {
                for index_tuple in poly.0 {
                    let positon_index = index_tuple.0;
                    let texcoord_index = index_tuple.1.unwrap();

                    let position = obj.data.position[positon_index];
                    let texcoord = obj.data.texture[texcoord_index];
                    let vertex = Vertex::new(
                        Vec3::from_array(position),
                        Vec3::new(1., 1., 1.),
                        Vec2::new(texcoord[0], 1. - texcoord[1]),
                    );

                    if deduplicate {
                        let key = (positon_index, texcoord_index);

                        if !vertex_hashmap.contains_key(&key) {
                            vertex_hashmap.insert(key, vertices.len() as u32);
                            vertices.push(vertex);
                        }

                        indices.push(*vertex_hashmap.get(&key).unwrap());
                    } else {
                        vertices.push(vertex);
                        indices.push(indices.len() as u32);
                    }
                }
            }
        }
    }

    println!("vertex count: {}", vertices.len());

    (vertices, indices)
}
