#shader vertex
#version 400 core
			
layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in vec4 a_Color;
layout(location = 3) in float a_TexIndex;
layout(location = 4) in float a_TilingFactor;
layout(location = 5) in mat4 a_Transform;

uniform mat4 u_ViewProjection;

out vec2 v_TexCoord;
out vec4 v_Color;
out float v_TexIndex;
out float v_TilingFactor;

void main()
{
	v_TexCoord = a_TexCoord;
	v_Color = a_Color;
	gl_Position = u_ViewProjection * a_Transform * vec4(a_Position, 1.0);	
}

#shader fragment
#version 400 core

layout(location = 0) out vec4 color;
in vec4 v_Color;
in vec2 v_TexCoord;

void main()
{
    vec2 p = v_TexCoord * 2.0 - 1.0;
    float d = length(p);

    // Edge width in screen space
    float aa = fwidth(d);

    // Radii
    float outerRadius = 0.9;
    float outlineRadius = 1.0;
    float innerRadius = clamp(v_Color.a, 0.1, 0.9);

    // Smooth masks
    float outer = 1.0 - smoothstep(outerRadius - aa, outerRadius + aa, d);
    float outline = (1.0 - smoothstep(outlineRadius - aa, outlineRadius + aa, d))
                - outer;
    float inner = 1.0 - smoothstep(innerRadius - aa, innerRadius + aa, d);

    // Colors
    vec3 rgb = mix(vec3(0.1), v_Color.rgb, inner);
    rgb = mix(rgb, vec3(0.8, 0.9, 0.8), outline);

    float alpha = max(outer, outline);

    color = vec4(rgb, alpha);
}
