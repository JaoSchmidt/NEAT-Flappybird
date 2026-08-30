#shader vertex
#version 400 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in vec4 a_Color;
layout(location = 3) in float a_TexIndex;
layout(location = 4) in float a_TilingFactor;

uniform mat4 u_ViewProjection;

out vec2 v_TexCoord;
out vec4 v_Color;
out float v_TexIndex;
out float v_TilingFactor;

void main()
{
    v_TexCoord = a_TexCoord;
    v_TexIndex = a_TexIndex;
    v_TilingFactor = a_TilingFactor;
    v_Color = a_Color;
    gl_Position = u_ViewProjection * vec4(a_Position, 1.0);
}

#shader fragment
#version 400 core

layout(location = 0) out vec4 color;

in vec2 v_TexCoord;
in vec4 v_Color;
in float v_TexIndex;
in float v_TilingFactor;

uniform sampler2D u_Textures[32];

//vec4 u_OuterColor = vec4(4/15, 4/15, 4/15, 1.0);
vec4 u_OuterColor = vec4(0.266666, 0.266666, 0.266666, 1.0);

float radius = 0.08;
float u_Thickness = 0.01;
float u_Smoothness = 0.005;
vec2 holeSize = vec2(1.5, 1);

float roundedBoxSDF(vec2 pos, vec2 b, float r)
{
    vec2 q = abs(pos) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main()
{
    // WARN: this needs to be equal to the rectangle size at all times, otherwise hole lines will have deformed thickness
    vec2 objectSize = vec2(10, 10); //

    vec2 p = (v_TexCoord - 0.5) * objectSize;
    
    float d = roundedBoxSDF(p, holeSize, radius);

    // Transparent hole
    if (d < -u_Thickness)
        discard;

    // Blend border -> outer region smoothly
    float t = smoothstep(-u_Smoothness, u_Smoothness, d);

    // Border uses v_Color
    vec4 c = mix(texture(u_Textures[int(v_TexIndex)], v_TexCoord * v_TilingFactor) * v_Color, u_OuterColor, t);

    color = c;
}
