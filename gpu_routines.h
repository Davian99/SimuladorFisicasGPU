#include "main.h"
#include "list_render_object.h"

class GPU{
	private:
		ListRenderObject * lro;
	public:
		GPU();
		void initializeContext();
};