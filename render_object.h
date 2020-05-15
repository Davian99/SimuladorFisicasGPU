#pragma once

class RenderObject {
	public:
		virtual void render() = 0;	
		virtual void setStatic() = 0;
		virtual void applyImpulse(float impulse_x, float impulse_y, float cvx, float cvy) = 0;

		float px, py, radius;
		float vx, vy;
		float force_x, force_y;
		float angularVelocity;
		float mass, inv_mass;
		float inertia, inv_inertia;
};

class Circle: public RenderObject {
	private:

	public:
		Circle(float x, float y, float r, bool _static);
		Circle();
		~Circle();
		void render() override;
		void setStatic() override;
		void applyImpulse(float impulse_x, float impulse_y, float cvx, float cvy) override;		
};
