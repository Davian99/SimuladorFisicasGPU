#pragma once

class RenderObject {
	public:
		virtual void render() = 0;	
		virtual void rotate1Deg() = 0;
		virtual void setStatic() = 0;
		virtual void applyForce(const float fx, const float fy) = 0;
		virtual void modifyPos(float x, float y) = 0;
		virtual void applyImpulse(float impulse_x, float impulse_y, float cvx, float cvy) = 0;
		virtual void setOrient(float radians) = 0;

		float px, py, real_r, radius;
		float vx, vy;
		float force_x, force_y;
		float angularVelocity, torque, orient;
		float mass, inv_mass;
		float inertia, inv_inertia;
		float staticFriction, dynamicFriction, restitution;
};

class Circle: public RenderObject {
	private:

	public:
		Circle(float x, float y, float r, bool _static);
		Circle();
		~Circle();
		void render() override;
		void rotate1Deg() override;
		void setStatic() override;
		void applyForce(const float fx, const float fy) override;
		void modifyPos(float x, float y) override;
		void applyImpulse(float impulse_x, float impulse_y, float cvx, float cvy) override;
		void setOrient(float radians) override;
		
};
