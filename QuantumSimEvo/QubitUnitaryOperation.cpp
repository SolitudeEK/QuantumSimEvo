#include "QubitUnitaryOperation.h"

    Gate2x2 QubitUnitaryOperation::getIdentity() {
        return Gate2x2(
            Complex(1.0), Complex(0.0),
            Complex(0.0), Complex(1.0)
        );
    }

    Gate2x2 QubitUnitaryOperation::getPauliX() {
        return Gate2x2(
            Complex(0.0), Complex(1.0),
            Complex(1.0), Complex(0.0)
        );
    }

    Gate2x2 QubitUnitaryOperation::getPauliY() {
        return Gate2x2(
            Complex(0.0), -I,
            I, Complex(0.0)
        );
    }

    Gate2x2 QubitUnitaryOperation::getPauliZ() {
        return Gate2x2(
            Complex(1.0), Complex(0.0),
            Complex(0.0), Complex(-1.0)
        );
    }

    Gate2x2 QubitUnitaryOperation::getHadamard() {
        Complex c(1.0 / std::sqrt(2.0));
        return Gate2x2(c, c, c, -c);
    }

    Gate2x2 QubitUnitaryOperation::getPhase(double theta) {
        // e^(i*theta) = cos(theta) + i*sin(theta)
        Complex phase = std::exp(I * theta);
        return Gate2x2(
            Complex(1.0), Complex(0.0),
            Complex(0.0), phase
        );
    }

    Gate2x2 QubitUnitaryOperation::getRotateX(double theta) {
        double cos = std::cos(theta / 2.0);
        double sin = std::sin(theta / 2.0);
        // -I * s is a Complex; explicit real parts use Complex(c, 0.0)
        return Gate2x2(
            Complex(cos), -I * sin,
            -I * sin, Complex(cos)
        );
    }

    Gate2x2 QubitUnitaryOperation::getRotateY(double theta) {
        double cos = std::cos(theta / 2.0);
        double sin = std::sin(theta / 2.0);
        return Gate2x2(
            Complex(cos), Complex(-sin),
            Complex(sin), Complex(cos)
        );
    }

    Gate2x2 QubitUnitaryOperation::getRotateZ(double theta) {
        Complex pos = std::exp(I * (theta / 2.0));
        Complex neg = std::exp(-I * (theta / 2.0));
        return Gate2x2(
            neg, Complex(0.0),
            Complex(0.0), pos
        );
    }
