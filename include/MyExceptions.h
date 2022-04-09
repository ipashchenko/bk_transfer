#ifndef MHD_TRANSFER_INCLUDE_MYEXCEPTIONS_H
#define MHD_TRANSFER_INCLUDE_MYEXCEPTIONS_H

#include <exception>


class AbsentDataFile : public std::exception {
    std::string msg;
    public:
        AbsentDataFile(const std::string& fname) : msg(std::string("Absent data file : ") + fname) {}
        const char * what () const noexcept override {
            return msg.c_str();
        }
};


class NotImplmentedParticlesHeating : public std::exception {
        std::string msg;
    public:
        NotImplmentedParticlesHeating(const std::string& model) : msg(std::string("Not implemented particles heating model: ") + model) {}
        const char * what () const noexcept override {
            return msg.c_str();
        }
};


class NotImplmentedHeatingSuppression : public std::exception {
        std::string msg;
    public:
        NotImplmentedHeatingSuppression(const std::string& model) : msg(std::string("Not implemented suppresion of particles heating: ") + model) {}
        const char * what () const noexcept override {
            return msg.c_str();
        }
};

class PhysicalException : public std::exception {
        std::string msg;
    public:
        PhysicalException(const std::string& quantity) : msg(std::string("Problems with quantity : ") + quantity) {}
        const char * what () const noexcept override {
            return msg.c_str();
        }
};

class BadParticleProfileParameters : public std::exception {
    const char * what () const noexcept override {
        return "Check particle profile parameters!";
    }
};


class NotImplmentedEquipartitionAnisotropicPowerLaw : public std::exception {
        const char * what () const noexcept override {
            return "Equipartition Power Law with anisotropic spectral index is not implemented!";
        }
};

class BadPlasmaContentParameters : public std::exception {
    const char * what () const noexcept override {
        return "Check plasma content parameters!";
    }
};

class PointOutsideOfPolyhedron : public std::exception {
    const char * what () const noexcept override {
        return "Point has z-coordinate that lies outside of Simulation Geometry Polyhedron!";
    }
};

class NaNStokesComponents : public std::exception {
        const char * what () const noexcept override {
            return "Some of the Stokes are NaN during RT!";
        }
};

class BadHeatingProfileParameters : public std::exception {
        const char * what () const noexcept override {
            return "Check parameters of the heating profile!";
        }
};

class NegativeKI : public std::exception {
        const char * what () const noexcept override {
            return "k_I < 0!";
        }
};


class NegativeTau : public std::exception {
        const char * what () const noexcept override {
            return "tau < 0!";
        }
};


class NegativeI : public std::exception {
        const char * what () const noexcept override {
            return "I < 0!";
        }
};


class NaNResult : public std::exception {
        const char * what () const noexcept override {
            return "NaN result!";
        }
};


#endif //MHD_TRANSFER_INCLUDE_MYEXCEPTIONS_H
