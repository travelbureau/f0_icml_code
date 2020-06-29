#include "gjk.hpp"
vec2 subtract(vec2 &a, vec2 &b) {
    vec2 sub;
    sub.x = a.x - b.x;
    sub.y = a.y - b.y;
    return sub;
}
vec2 negate(vec2 &v) {
    vec2 v_n;
    v_n.x = -v.x;
    v_n.y = -v.y;
    return v_n;
}
vec2 perpendicular(vec2 &v) {
    vec2 p = {v.y, -v.x};
    return p;
}
double dotProduct(vec2 &a, vec2 &b) {
    return a.x*b.x+a.y*b.y;
}
double lengthSquared(vec2 &v) {
    return v.x*v.x + v.y*v.y;
}
vec2 tripleProduct(vec2 &a, vec2 &b, vec2 &c) {
    vec2 r;
    double ac = a.x*c.x + a.y*c.y;
    double bc = b.x*c.x + b.y*c.y;
    r.x = b.x*ac - a.x*bc;
    r.y = b.y*ac - a.y*bc;
    return r;
}
vec2 averagePoint(std::vector<vec2> &vertices) {
    size_t count = vertices.size();
    vec2 avg = {0., 0.};
    for (size_t i=0; i<count; i++) {
        avg.x += vertices[i].x;
        avg.y += vertices[i].y;
    }
    avg.x /= count;
    avg.y /= count;
    return avg;
}
size_t indexOfFurthestPoint(std::vector<vec2> &vertices, vec2 &d) {
    size_t count = vertices.size();
    double maxProduct = dotProduct(d, vertices[0]);
    size_t index = 0;
    for (size_t i=1; i<count; i++) {
        double product = dotProduct(d, vertices[i]);
        if (product > maxProduct) {
            maxProduct = product;
            index = i;
        }
    }
    return index;
}
vec2 support(std::vector<vec2> &vertices1, std::vector<vec2> &vertices2, vec2 &d) {
    vec2 d_n = negate(d);
    size_t i = indexOfFurthestPoint(vertices1, d);
    size_t j = indexOfFurthestPoint(vertices2, d_n);
    vec2 support = subtract(vertices1[i], vertices2[j]);
    return support;
}
int gjk(std::vector<vec2> &vertices1, std::vector<vec2> &vertices2) {
    size_t index = 0;
    vec2 a, b, c, d, ao, ab, ac, abperp, acperp;
    std::vector<vec2> simplex(3);
    vec2 position1 = averagePoint(vertices1);
    vec2 position2 = averagePoint(vertices2);
    d = subtract(position1, position2);
    if ((d.x == 0) && (d.y == 0)) {
        d.x = 1.;
    }
    a = support(vertices1, vertices2, d);
    simplex[0] = a;
    if (dotProduct(a, d) <= 0) {
        return 0;
    }
    d = negate(a);
    int iter_count = 0;
    while (1) {
        iter_count++;
        a = support(vertices1, vertices2, d);
        simplex[++index] = a;
        if (dotProduct(a, d) <= 0) {
            return 0;
        }
        ao = negate(a);
        if (index < 2) {
            b = simplex[0];
            ab = subtract(b, a);
            d = tripleProduct(ab, ao, ab);
            if (lengthSquared(d) == 0) {
                d = perpendicular(ab);
            }
            continue;
        }
        b = simplex[1];
        c = simplex[0];
        ab = subtract(b, a);
        ac = subtract(c, a);
        acperp = tripleProduct(ab, ac, ac);
        if (dotProduct(acperp, ao) >= 0) {
            d = acperp;
        } else {
            abperp = tripleProduct(ac, ab, ab);
            if (dotProduct(abperp, ao) < 0) {
                return 1;
            }
            simplex[0] = simplex[1];
            d = abperp;
        }
        simplex[1] = simplex[2];
        --index;
    }
    return 0;
}
double Perturbation();
vec2 Jostle(vec2 a);
