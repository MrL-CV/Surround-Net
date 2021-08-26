//多边形IoU的计算

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;
#define maxn 51
const double eps = 1E-8;

//如果d落在[-eps,eps]中间,return0;反之return1/-1
int sig(double d) {
    return (d > eps) - (d < -eps);
}

//结构体Point
struct Point {
    // 数据成员
	double x, y;
	// 默认构造函数
    Point() {}
	// 双参数构造函数
    Point(double x, double y) : x(x), y(y) {}
	// 判断是否与另外某个点坐标重合
    bool operator==(const Point &p) const {
        return sig(x - p.x) == 0 && sig(y - p.y) == 0;
    }
};

// 凸包面积计算
// 几何意义:oa向量与ob向量构成的平行四边形的面积
double cross(Point o, Point a, Point b) {  //叉积
    return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}

// 点集的面积:两两之间点构成的平行四边形面积的总和/2
// ps:顶点坐标数组,n:多边形边数
double area(Point *ps, int n) {
    ps[n] = ps[0];
    double res = 0;
    for (int i = 0; i < n; i++) {
        res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
    }
    return res / 2.0;
}

// 计算两条直线交点:a->b和c->d
int lineCross(Point a, Point b, Point c, Point d, Point &p) {
    double s1, s2;
    s1 = cross(a, b, c); // 三角形abc面积*2
    s2 = cross(a, b, d); // 三角形abd面积*2
    if (sig(s1) == 0 && sig(s2) == 0) return 2; // 两个面积同时都很小
    if (sig(s2 - s1) == 0) return 0; // 两者面积类型相等
	// 面积类型不相等,说明有交点,交点计算公式为:
    p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
    p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
    return 1; // 否则返回1
}

// 使用ab去切割多边形,因为多次调用这个函数,所以切割的线总是在变换
void polygon_cut(Point *p, int &n, Point a, Point b, Point *pp) {
//    static Point pp[maxn];
    int m = 0; // 
    p[n] = p[0]; // 形成1个环
    for (int i = 0; i < n; i++) {
		// 如果a-b-pi的面积足够大,且方向为正
        if (sig(cross(a, b, p[i])) > 0) pp[m++] = p[i];
		// 如果a-b-pi的面积类型 != a-b-pi+1的面积类型
		if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1])))
			// 0/1, 0/-1, -1/1
            lineCross(a, b, p[i], p[i + 1], pp[m++]);
    }
    n = 0;
    for (int i = 0; i < m; i++)
        if (!i || !(pp[i] == pp[i - 1]))
            p[n++] = pp[i];
    while (n > 1 && p[n - 1] == p[0])n--;
}

//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
double intersectArea(Point a, Point b, Point c, Point d) {
    Point o(0, 0); // 确定原点
    int s1 = sig(cross(o, a, b)); // 三角形oab面积*2的面积类型
    int s2 = sig(cross(o, c, d)); // 三角形ocd面积*2的面积类型
    if (s1 == 0 || s2 == 0)return 0.0; // 如果2者面积都很小,交集=0
	// 将面积变为正数,调转读取方向
    if (s1 == -1) swap(a, b);
    if (s2 == -1) swap(c, d);

	// 
    Point p[10] = {o, a, b}; 
    int n = 3;
    Point pp[maxn];
	// 用oc-cd-do3条线去切割另一个三角形OAB,
    polygon_cut(p, n, o, c, pp);
    polygon_cut(p, n, c, d, pp);
    polygon_cut(p, n, d, o, pp);
    double res = fabs(area(p, n));
    if (s1 * s2 == -1) res = -res; // 若2个三角形读取点的顺序是否一致,如果一致就是正的,否则是负的
    return res;
}

//求两多边形的交面积
double intersectArea(Point *ps1, int n1, Point *ps2, int n2) { // ps1是多边形1的所有顶点坐标,共n1个点; ps2是多边形2的所有顶点坐标,共n2个点;
    if (area(ps1, n1) < 0) reverse(ps1, ps1 + n1);
    if (area(ps2, n2) < 0) reverse(ps2, ps2 + n2);
    ps1[n1] = ps1[0];
    ps2[n2] = ps2[0];
    double res = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
			// 有正有负,最后总是正
            res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
        }
    }
    return res;//assume res is positive!
}

// 输出IoU
// 参数:2个多边形的顶点
double iou_poly(vector<double> p, vector<double> q) {
    Point ps1[maxn], ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    double inter_area = intersectArea(ps1, n1, ps2, n2);
    double union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    double iou = inter_area / union_area;

//    cout << "inter_area:" << inter_area << endl;
//    cout << "union_area:" << union_area << endl;
//    cout << "iou:" << iou << endl;

    return iou;
}
//
//int main(){
//    double p[8] = {0, 0, 1, 0, 1, 1, 0, 1};
//    double q[8] = {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5};
//    vector<double> P(p, p + 8);
//    vector<double> Q(q, q + 8);
//    iou_poly(P, Q);
//    return 0;
//}

//int main(){
//    double p[8] = {0, 0, 1, 0, 1, 1, 0, 1};
//    double q[8] = {0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5};
//    iou_poly(p, q);
//    return 0;
//}
