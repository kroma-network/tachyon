use std::{
    io::{Read, Result},
    mem::size_of,
};

fn read_vec<T, R>(reader: &mut R) -> Result<Vec<T>>
where
    T: Readable,
    R: Read,
{
    let mut len = [0u8; size_of::<usize>()];
    reader.read_exact(&mut len)?;

    let len = usize::from_ne_bytes(len);

    (0..len)
        .map(|_| T::read_from(reader))
        .collect::<Result<Vec<_>>>()
}

pub trait Readable: Sized {
    fn read_from<R: Read>(reader: &mut R) -> Result<Self>;
}

impl<T: Readable> Readable for Vec<T> {
    fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        read_vec(reader)
    }
}

impl<const D: usize> Readable for [u32; D] {
    fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let data = [0u32; D];
        Ok(unsafe {
            let mut slice =
                std::slice::from_raw_parts_mut(data.as_ptr() as *mut u8, data.len() * 4);
            reader.read_exact(&mut slice)?;
            std::mem::transmute(data)
        })
    }
}
