"""What a generator hands back: arrays, named the way glTF names them.

glTF supplies the *vocabulary* here, not the container. The attributes are called
``POSITION``, ``NORMAL``, ``TEXCOORD_0`` and so on because those names are
understood everywhere, and they are stored in the types glTF stores them in --
but they are stored as plain NumPy arrays, not as accessors into a binary blob.
Almost nothing that generates geometry goes on to write a file, and a caller who
had to decode an accessor to reach a vertex would be worse off than one handed
the array.

So the everyday path is:

    >>> mesh = extrude(circle(0.2), path=spine)             # doctest: +SKIP
    >>> mesh.primitives[0].positions                        # doctest: +SKIP
    array([...], dtype=float32)

and writing a file is a side door, for the cases that want one:
:meth:`Mesh.to_gltf` and :meth:`Mesh.to_glb`.

**The arrays are ready to upload, in the form a glTF renderer already uses.**
Attributes are C-contiguous ``float32`` and indices are ``uint32``. That is the
same arrangement OpenGLContext's ``PBRMesh`` holds -- the node its glTF loader
builds for every primitive of every ``.glb`` it reads -- so a generated mesh and
a decoded one reach the renderer indistinguishable from one another, and
:meth:`Primitive.arrays` renames the semantics to the keywords that node takes.
Passing a mesh on therefore costs nothing: there is nothing left to convert, and
an array handed in already in that form is kept rather than copied.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from opengl_extrusions import weld as _weld
from opengl_extrusions.types import Points

__all__ = ['Mesh', 'Primitive', 'MeshError', 'ATTRIBUTE_WIDTHS']

#: How wide each attribute this library produces is, and the glTF accessor type
#: that goes with it.
ATTRIBUTE_WIDTHS: dict[str, tuple[int, str]] = {
    'POSITION': (3, 'VEC3'),
    'NORMAL': (3, 'VEC3'),
    'TANGENT': (4, 'VEC4'),
    'TEXCOORD_0': (2, 'VEC2'),
    'TEXCOORD_1': (2, 'VEC2'),
    'COLOR_0': (4, 'VEC4'),
}

#: The renderer-facing keyword each attribute maps to in :meth:`Primitive.arrays`.
_KEYWORDS = {
    'POSITION': 'positions',
    'NORMAL': 'normals',
    'TEXCOORD_0': 'texcoords',
    'TEXCOORD_1': 'texcoords1',
    'TANGENT': 'tangents',
    'COLOR_0': 'colors',
}

_GL_FLOAT = 5126
_GL_UNSIGNED_INT = 5125
_GL_ARRAY_BUFFER = 34962
_GL_ELEMENT_ARRAY_BUFFER = 34963
_MODE_TRIANGLES = 4


class MeshError(ValueError):
    """A mesh does not describe geometry that can be drawn."""


def _as_float32(array: Any) -> np.ndarray:
    """A C-contiguous float32 view, without copying one that already is."""
    out = np.asarray(array, dtype=np.float32)
    return np.ascontiguousarray(out)


@dataclass
class Primitive:
    """One drawable batch: vertex attributes, and optionally an index buffer.

    ``attributes`` maps glTF semantic names to arrays. ``POSITION`` is required;
    everything else is optional and present when the generator had something to
    say. ``indices`` is a flat ``uint32`` array of triangle corners, or ``None``
    for a mesh drawn straight from its vertex order.

    ``extras`` records what produced the primitive and with which parameters, so
    a mesh written to a file says where it came from.
    """

    attributes: dict[str, np.ndarray]
    indices: np.ndarray | None = None
    mode: int = _MODE_TRIANGLES
    material: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.attributes = {name: _as_float32(value) for name, value in self.attributes.items()}
        if self.indices is not None:
            self.indices = np.ascontiguousarray(np.asarray(self.indices, dtype=np.uint32).ravel())

    # -- named access -----------------------------------------------------

    @property
    def positions(self) -> np.ndarray:
        return self.attributes['POSITION']

    @property
    def normals(self) -> np.ndarray | None:
        return self.attributes.get('NORMAL')

    @property
    def texcoords(self) -> np.ndarray | None:
        return self.attributes.get('TEXCOORD_0')

    @property
    def tangents(self) -> np.ndarray | None:
        return self.attributes.get('TANGENT')

    @property
    def colors(self) -> np.ndarray | None:
        return self.attributes.get('COLOR_0')

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def triangle_count(self) -> int:
        if self.indices is not None:
            return len(self.indices) // 3
        return self.vertex_count // 3

    @property
    def triangles(self) -> np.ndarray:
        """The index buffer as ``(T, 3)``, whether or not one was supplied."""
        if self.indices is None:
            return np.arange(self.vertex_count, dtype=np.uint32)[: self.triangle_count * 3].reshape(
                -1, 3
            )
        return self.indices[: self.triangle_count * 3].reshape(-1, 3)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(minimum, maximum)`` corner of the axis-aligned bounding box."""
        if self.vertex_count == 0:
            zero = np.zeros(3, dtype=np.float32)
            return zero, zero.copy()
        return self.positions.min(axis=0), self.positions.max(axis=0)

    def arrays(self) -> dict[str, np.ndarray]:
        """The attributes under the keyword names a renderer node expects.

        Built for handing straight to a mesh node::

            PBRMesh(**primitive.arrays())
        """
        out = {
            _KEYWORDS[name]: value for name, value in self.attributes.items() if name in _KEYWORDS
        }
        if self.indices is not None:
            out['indices'] = self.indices
        return out

    # -- checks -----------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`MeshError` unless this primitive can actually be drawn.

        Checks that positions exist and are finite, that every attribute is the
        right width and the same length as the positions, and that the indices
        are whole triangles pointing at vertices that exist.
        """
        if 'POSITION' not in self.attributes:
            raise MeshError('primitive has no POSITION attribute')
        count = self.vertex_count
        for name, value in self.attributes.items():
            expected = ATTRIBUTE_WIDTHS.get(name, (value.shape[-1], ''))[0]
            if value.ndim != 2 or value.shape[1] != expected:
                raise MeshError(
                    'attribute %s should be (N, %d), got %r' % (name, expected, value.shape)
                )
            if len(value) != count:
                raise MeshError(
                    'attribute %s has %d vertices, POSITION has %d' % (name, len(value), count)
                )
        if not np.isfinite(self.positions).all():
            raise MeshError('POSITION contains a non-finite coordinate')
        if self.indices is not None:
            if len(self.indices) % 3 and self.mode == _MODE_TRIANGLES:
                raise MeshError(
                    'index count %d is not a whole number of triangles' % len(self.indices)
                )
            if len(self.indices) and int(self.indices.max()) >= count:
                raise MeshError(
                    'index %d refers to a vertex beyond the %d present'
                    % (int(self.indices.max()), count)
                )

    def _position_topology(self, tolerance: float = 0.0) -> np.ndarray:
        """The triangles with vertices at the same place treated as the same vertex.

        Both topology questions are about the *surface*, and a surface is not
        open just because a seam's vertices are duplicated to carry two normals
        or two texture coordinates -- which they almost always are, since that is
        what a hard edge means.
        """
        _, mapping = _weld.weld_vertices(self.positions, tolerance=tolerance)
        return mapping[self.triangles.ravel()].reshape(-1, 3)

    def is_manifold(self, tolerance: float = 0.0) -> bool:
        """Whether no edge of this surface is shared by more than two triangles.

        Vertices at the same position count as the same vertex.
        """
        return _weld.is_manifold(self._position_topology(tolerance))

    def is_watertight(self, tolerance: float = 0.0) -> bool:
        """Whether this surface is closed: no rim, no gaps, nothing to leak out.

        Vertices at the same position count as the same vertex, so a shape whose
        seams are split for shading still reports closed -- which is the question
        a caller is asking when they want to know whether a solid is solid.
        """
        return _weld.is_watertight(self._position_topology(tolerance))

    def signed_volume(self) -> float:
        """Volume enclosed, if the surface is closed; negative if inside out."""
        return _weld.signed_volume(self.positions, self.triangles)

    def surface_area(self) -> float:
        """Total area of the triangles."""
        return _weld.surface_area(self.positions, self.triangles)

    # -- derived primitives -----------------------------------------------

    def welded(self, tolerance: float = 0.0) -> Primitive:
        """A copy with duplicate vertices merged and the indices rewritten.

        Vertices merge only when every attribute agrees, so a hard edge stays
        hard: two corners at the same place facing different ways remain two.
        """
        extra = [value for name, value in sorted(self.attributes.items()) if name != 'POSITION']
        order, mapping = _weld.weld_vertices(self.positions, extra, tolerance)
        attributes = {
            name: np.ascontiguousarray(value[order]) for name, value in self.attributes.items()
        }
        indices = mapping[self.triangles.ravel()].astype(np.uint32)
        return Primitive(attributes, indices, self.mode, self.material, dict(self.extras))

    def transformed(self, matrix: Points) -> Primitive:
        """A copy moved by a 4x4 matrix.

        Positions go through the matrix; normals and tangents go through the
        inverse transpose of its rotation part, which is what keeps them
        perpendicular to the surface under a non-uniform scale, and are
        re-normalised afterwards.
        """
        m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        attributes = dict(self.attributes)
        points = np.column_stack([self.positions.astype(np.float64), np.ones(self.vertex_count)])
        attributes['POSITION'] = _as_float32((points @ m.T)[:, :3])
        normal_matrix = np.linalg.inv(m[:3, :3]).T
        for name in ('NORMAL', 'TANGENT'):
            value = self.attributes.get(name)
            if value is None:
                continue
            vectors = value[:, :3].astype(np.float64) @ normal_matrix.T
            lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)
            if name == 'TANGENT':
                attributes[name] = _as_float32(np.column_stack([vectors, value[:, 3]]))
            else:
                attributes[name] = _as_float32(vectors)
        return Primitive(
            attributes,
            None if self.indices is None else self.indices.copy(),
            self.mode,
            self.material,
            dict(self.extras),
        )

    def reversed(self) -> Primitive:
        """A copy facing the other way: winding flipped, normals negated."""
        attributes = dict(self.attributes)
        for name in ('NORMAL',):
            if name in attributes:
                attributes[name] = _as_float32(-attributes[name])
        # Swap the last two corners rather than reversing all three: the winding
        # flips either way, but this keeps each triangle's first vertex, which is
        # the one a flat-shading renderer takes its attributes from.
        indices = self.triangles[:, [0, 2, 1]].ravel().astype(np.uint32)
        return Primitive(
            attributes, np.ascontiguousarray(indices), self.mode, self.material, dict(self.extras)
        )

    def layout(self) -> tuple[str, ...]:
        """The attribute names present, sorted: two primitives with the same
        layout can be concatenated."""
        return tuple(sorted(self.attributes))


@dataclass
class Mesh:
    """A group of primitives, and the things one does with a whole shape.

    A generator returns one of these. Callers customise by editing the arrays in
    place, appending primitives, adding meshes together, or assigning materials,
    then either hand it to a renderer or serialise it.
    """

    primitives: list[Primitive] = field(default_factory=list)
    name: str | None = None

    def __len__(self) -> int:
        return len(self.primitives)

    def __add__(self, other: Mesh) -> Mesh:
        if not isinstance(other, Mesh):
            return NotImplemented
        return Mesh(list(self.primitives) + list(other.primitives), self.name)

    @property
    def vertex_count(self) -> int:
        return sum(p.vertex_count for p in self.primitives)

    @property
    def triangle_count(self) -> int:
        return sum(p.triangle_count for p in self.primitives)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(minimum, maximum)`` corner of the box around every primitive."""
        boxes = [p.bounds for p in self.primitives if p.vertex_count]
        if not boxes:
            zero = np.zeros(3, dtype=np.float32)
            return zero, zero.copy()
        low = np.min([b[0] for b in boxes], axis=0)
        high = np.max([b[1] for b in boxes], axis=0)
        return low, high

    def validate(self) -> None:
        """Check every primitive; raise :class:`MeshError` on the first fault."""
        for p in self.primitives:
            p.validate()

    def merged(self) -> Mesh:
        """A copy with primitives sharing a layout and material concatenated.

        One draw call instead of several, at the cost of losing the boundary
        between them. Primitives whose attributes differ cannot be concatenated
        and are left as they are.
        """
        groups: dict[tuple[Any, ...], list[Primitive]] = {}
        order: list[tuple[Any, ...]] = []
        for p in self.primitives:
            key = (p.layout(), p.mode, p.material)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(p)

        out: list[Primitive] = []
        for key in order:
            members = groups[key]
            if len(members) == 1:
                out.append(members[0])
                continue
            attributes = {
                name: np.concatenate([p.attributes[name] for p in members])
                for name in members[0].attributes
            }
            indices: list[np.ndarray] = []
            offset = 0
            for p in members:
                indices.append(p.triangles.ravel().astype(np.uint32) + offset)
                offset += p.vertex_count
            out.append(
                Primitive(
                    attributes, np.concatenate(indices), key[1], key[2], {'merged': len(members)}
                )
            )
        return Mesh(out, self.name)

    def welded(self, tolerance: float = 0.0) -> Mesh:
        """A copy with each primitive's duplicate vertices merged."""
        return Mesh([p.welded(tolerance) for p in self.primitives], self.name)

    def transformed(self, matrix: Points) -> Mesh:
        """A copy moved by a 4x4 matrix."""
        return Mesh([p.transformed(matrix) for p in self.primitives], self.name)

    def reversed(self) -> Mesh:
        """A copy facing the other way: every primitive's winding and normals flipped."""
        return Mesh([p.reversed() for p in self.primitives], self.name)

    # -- serialisation ----------------------------------------------------

    def to_gltf(self, embed: bool = True) -> dict[str, Any]:
        """A glTF 2.0 document as a plain dictionary.

        With ``embed`` the buffer is a base64 data URI, so the document is
        self-contained and ``json.dumps`` writes a complete ``.gltf`` file.
        Without it the buffer is described but its bytes are left for the caller
        to place, which is what :meth:`to_glb` does.
        """
        blob = bytearray()
        views: list[dict[str, Any]] = []
        accessors: list[dict[str, Any]] = []
        meshes: list[dict[str, Any]] = []
        primitives: list[dict[str, Any]] = []

        for p in self.primitives:
            entry: dict[str, Any] = {'attributes': {}, 'mode': p.mode}
            for name, value in sorted(p.attributes.items()):
                entry['attributes'][name] = _append_accessor(
                    blob,
                    views,
                    accessors,
                    value,
                    name,
                    _GL_ARRAY_BUFFER,
                    _GL_FLOAT,
                    ATTRIBUTE_WIDTHS.get(name, (value.shape[1], 'VEC%d' % value.shape[1]))[1],
                )
            if p.indices is not None:
                entry['indices'] = _append_accessor(
                    blob,
                    views,
                    accessors,
                    p.indices.reshape(-1, 1),
                    'indices',
                    _GL_ELEMENT_ARRAY_BUFFER,
                    _GL_UNSIGNED_INT,
                    'SCALAR',
                )
            if p.material is not None:
                entry['material'] = p.material
            if p.extras:
                entry['extras'] = _jsonable(p.extras)
            primitives.append(entry)

        if primitives:
            meshes.append({'primitives': primitives, **({'name': self.name} if self.name else {})})

        document: dict[str, Any] = {
            'asset': {'version': '2.0', 'generator': 'opengl_extrusions'},
            'scene': 0,
            'scenes': [{'nodes': [0] if meshes else []}],
            'nodes': [{'mesh': 0}] if meshes else [],
            'meshes': meshes,
            'accessors': accessors,
            'bufferViews': views,
            'buffers': [],
        }
        if blob:
            buffer: dict[str, Any] = {'byteLength': len(blob)}
            if embed:
                buffer['uri'] = 'data:application/octet-stream;base64,' + base64.b64encode(
                    bytes(blob)
                ).decode('ascii')
            document['buffers'].append(buffer)
        document['_blob'] = bytes(blob) if not embed else b''
        if embed:
            del document['_blob']
        return document

    def to_glb_bytes(self) -> bytes:
        """The mesh as a binary glTF container, in memory.

        Useful for handing a complete asset to something that reads glTF without
        going near the filesystem.
        """
        document = self.to_gltf(embed=False)
        blob = document.pop('_blob', b'')
        json_chunk = json.dumps(document, separators=(',', ':')).encode('utf-8')
        json_chunk += b' ' * (-len(json_chunk) % 4)
        binary_chunk = blob + b'\x00' * (-len(blob) % 4)

        total = 12 + 8 + len(json_chunk) + (8 + len(binary_chunk) if binary_chunk else 0)
        out = bytearray()
        out += struct.pack('<4sII', b'glTF', 2, total)
        out += struct.pack('<II', len(json_chunk), 0x4E4F534A)
        out += json_chunk
        if binary_chunk:
            out += struct.pack('<II', len(binary_chunk), 0x004E4942)
            out += binary_chunk
        return bytes(out)

    def to_glb(self, path: str) -> None:
        """Write the mesh to a ``.glb`` file."""
        with open(path, 'wb') as handle:
            handle.write(self.to_glb_bytes())


def _append_accessor(
    blob: bytearray,
    views: list[dict[str, Any]],
    accessors: list[dict[str, Any]],
    array: np.ndarray,
    name: str,
    target: int,
    component_type: int,
    accessor_type: str,
) -> int:
    """Copy one array into the buffer and describe it, returning its index."""
    data = np.ascontiguousarray(array)
    blob += b'\x00' * (-len(blob) % 4)  # accessors must be 4-byte aligned
    offset = len(blob)
    blob += data.tobytes()
    views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': data.nbytes, 'target': target})
    accessor: dict[str, Any] = {
        'bufferView': len(views) - 1,
        'componentType': component_type,
        'count': len(data),
        'type': accessor_type,
        'name': name,
    }
    if accessor_type != 'SCALAR' and len(data):
        # POSITION's bounds are required by the specification; supplying them for
        # every attribute costs six numbers and saves a reader a pass.
        accessor['min'] = [float(v) for v in np.asarray(data).min(axis=0)]
        accessor['max'] = [float(v) for v in np.asarray(data).max(axis=0)]
    accessors.append(accessor)
    return len(accessors) - 1


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars and arrays so ``json.dumps`` accepts them."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
