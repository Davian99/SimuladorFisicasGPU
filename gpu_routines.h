#include "main.h"
#include "list_circles.h"

class GPU{
	private:
		int N_GPU_obj, MAX_GPU_obj;
		ListCircles * lro;
		Circle * circles_GPU;
	public:
		GPU(ListCircles * list);
		GPU();
		void initializeContext();
		void update_mem();
};
