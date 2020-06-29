#pragma once         
#include <iostream>
#include <vector>
struct _vec2{double x; double y;};
typedef struct _vec2 vec2;
vec2 subtract(vec2 &a, vec2 &b);
vec2 negate(vec2 &v);
vec2 perpendicular(vec2 &v);
double dotProduct(vec2 &a, vec2 &b);
double lengthSquared(vec2 &v);
vec2 tripleProduct(vec2 &a, vec2 &b, vec2 &c);
vec2 averagePoint(std::vector<vec2> &vertices);
size_t indexOfFurthestPoint(std::vector<vec2> &vertices, vec2 &d);
vec2 support(std::vector<vec2> &vertices1, std::vector<vec2> &vertices2, vec2 &d);
int gjk(std::vector<vec2> &vertices1, std::vector<vec2> &vertices2);
double Perturbation();
vec2 Jostle(vec2 &a);
