#version 450

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2) uniform vecTwo{
    vec2 xy;
} screenSize;

layout(location = 0) out vec4 outColor;

in layout(location = 0) vec3 vertcol;

void main() {
    vec2 fragTexCoord = vec2(gl_FragCoord.xy/screenSize.xy);
    vec3 pixel = texture(texSampler, fragTexCoord).rgb;

    outColor = vec4(pixel, 1.0);
    //utColor = vec4(gl_FragCoord.xy/800.0,0.0,1.0);
}