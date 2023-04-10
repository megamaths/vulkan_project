#version 450

layout(location = 0) out vec4 outColor;

in layout(location = 0) vec3 vertcol;

void main() {
    outColor = vec4(vertcol, 1.0);
}