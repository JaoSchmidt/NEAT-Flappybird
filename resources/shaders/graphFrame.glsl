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
	v_Color = a_Color;
	gl_Position = u_ViewProjection * vec4(a_Position, 1.0);	
}

#shader fragment
#version 400 core

layout(location = 0) out vec4 color;

in vec2 v_TexCoord;
in vec4 v_Color;

uniform vec4 u_OuterColor = vec4(0.5, 0.5, 0.5, 1.0);

uniform float u_Radius = 0.08;
uniform float u_BorderWidth = 0.03;
uniform float u_Smoothness = 0.002;

float roundedBoxSDF(vec2 p, vec2 b, float r)
{
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main()
{
    // Center UVs around (0,0)
    vec2 p = v_TexCoord - 0.5;

    // Half-size of the outer rounded rectangle
    vec2 size = vec2(0.45);

    float d = roundedBoxSDF(p, size, u_Radius);

    // Transparent hole
    if (d < -u_BorderWidth)
        discard;

    // Blend border -> outer region smoothly
    float t = smoothstep(-u_Smoothness, u_Smoothness, d);

    // Border uses v_Color, outside uses gray
    vec4 c = mix(v_Color, u_OuterColor, t);

    color = c;
}
