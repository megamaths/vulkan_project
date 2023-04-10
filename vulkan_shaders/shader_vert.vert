#version 450

in layout(location = 0)vec3 pos;
in layout(location = 1)vec3 col;
out layout(location = 0)vec3 vertcol;

void main() {
    vec3 outpos = vec3(pos[0]/pos[2],pos[1]/pos[2],0.5);
    gl_Position = vec4(outpos, 1.0);
    vertcol = col;
}