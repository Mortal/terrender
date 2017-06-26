#include <queue>
#include <set>
#include <tuple>

#include "rectangle_sweep.h"

namespace {

struct rect_order {
	bool operator()(const rect & a, const rect & b) const {
		return std::tie(a.x1, a.x2, a.y1, a.y2, a.idx) <
			std::tie(b.x1, b.x2, b.y1, b.y2, b.idx);
	}
};

bool intersects(const rect & a, const rect & b) {
	return a.x1 <= b.x2 && b.x1 <= a.x2 &&
		a.y1 <= b.y2 && b.y1 <= a.y2;
}

struct rect_y2_reverse_order {
	bool operator()(const rect & a, const rect & b) const {
		return (a.y2 != b.y2) ? (a.y2 > b.y2) : rect_order()(a, b);
	}
};

class rectangle_sweep {
public:
	rectangle_sweep() : seen(0) {}

	bool pop_until(rect a) {
		while (!exit_queue.empty() && exit_queue.top().y2 < a.y1) {
			size_t erased = sweep_line.erase(exit_queue.top());
			if (erased != 1)
				return false;
			exit_queue.pop();
		}
		return true;
	}

	long push(rect a, long * output_begin, long * output_end) {
		if (seen > 0 && a.y1 < prev_y) return -seen;
		seen += 1;
		prev_y = a.y1;

		if (!pop_until(a)) return -1;
		long i = 0;
		long n = output_end - output_begin;
		for (const auto & b : sweep_line)
			if (intersects(a, b)) {
				if (i == n) return -42;
				else output_begin[i++] = b.idx;
			}
		sweep_line.insert(a);
		exit_queue.push(a);
		return i;
	}

private:
	std::set<rect, rect_order> sweep_line;
	std::priority_queue<rect, std::vector<rect>, rect_y2_reverse_order> exit_queue;
	double prev_y;
	long seen;
};

}

extern "C" {

	void * rectangle_sweep_init() {
		return (void *) new rectangle_sweep;
	}

	long rectangle_sweep_push(void * o, double x1, double x2, double y1, double y2, long idx, long * o1, long * o2) {
		return ((rectangle_sweep *) o)->push({x1, x2, y1, y2, idx}, o1, o2);
	}

	void rectangle_sweep_free(void * o) {
		delete (rectangle_sweep *) o;
	}

}
