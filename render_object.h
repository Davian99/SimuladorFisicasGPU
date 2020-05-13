#pragma once

class RenderObject {
	public:
		virtual void render() = 0;	
		virtual void rotate1Deg() = 0;
		virtual void setStatic() = 0;
		virtual void ApplyForce(const float fx, const float fy) = 0;
		virtual void modifyPos(float x, float y) = 0;
};

class Circle: public RenderObject {
	private:
		
		float vx, vy;
		float force_x, force_y;
		float angularVelocity, torque, orient;
		float mass, inv_mass;
		float inertia, inv_inertia;
		float staticFriction, dynamicFriction, restitution;

	public:
		float cx, cy, radius; //TODO poner en private
		Circle(float x, float y, float r);
		Circle();
		~Circle();
		void render() override;
		void rotate1Deg() override;
		void setStatic() override;
		void ApplyForce(const float fx, const float fy) override;
		void modifyPos(float x, float y) override;
		
};