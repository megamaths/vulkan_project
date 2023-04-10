#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

in layout(location = 0)vec3 pos;
in layout(location = 1)vec3 col;
out layout(location = 0)vec3 vertcol;

void main() {
    gl_Position = ubo.proj*ubo.view*ubo.model*vec4(pos, 1.0);
    vertcol = col;
}