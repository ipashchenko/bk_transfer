import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters, report_fit
import corner
# emcee must be installed


def r_z(z):
    return np.sqrt(z)


# def kh_coordinates(z_proj, a, lam, ph, theta):
def kh_coordinates(z_proj, a, lam, ph):
    theta = np.deg2rad(17)
    z = z_proj/np.sin(theta)
    r0 = 1.0
    # x = a*r_z(z)/r0 * np.cos(2*np.pi*z / lam * r0 / r_z(z) + ph) * np.cos(theta) - z*np.sin(theta)
    y = a*r_z(z)/r0 * np.sin(2*np.pi*z / lam * r0 / r_z(z) + ph)
    return y


# # z_i - projected z
# data_1 = np.load('thread_1.npy')
# z_1, y_1 = data_1['x'], data_1['y']
# data_2 = np.load('thread_2.npy')
# z_2, y_2 = data_2['x'], data_2['y']
# fig, axes = plt.subplots(1, 1)
# axes.scatter(z_1, y_1)
# axes.scatter(z_2, y_2)
# axes.axis("equal")
# plt.show()


data_1 = np.load('thread_1.npy')
z_1, y_1 = data_1['x'], data_1['y']
data_2 = np.load('thread_2.npy')
z_2, y_2 = data_2['x'], data_2['y']

kh_model = Model(kh_coordinates)
params = Parameters()
params.add("a", value=1.0, min=0.1, max=2)
params.add("lam", value=20, min=3, max=50)
params.add("ph", value=1.0, min=0.0, max=2*np.pi)
# params.add("theta", value=np.deg2rad(20), min=np.deg2rad(10), max=np.deg2rad(30))
result_1 = kh_model.fit(y_1, params, z_proj=z_1)
result_2 = kh_model.fit(y_2, params, z_proj=z_2)

# Warning: here bounds are absent!
best_params_1 = kh_model.make_params(**result_1.best_values)
best_params_2 = kh_model.make_params(**result_2.best_values)
zz = np.linspace(min(np.min(z_1), np.min(z_2)), max(np.max(z_1), np.max(z_2)), 10000)
fig, axes = plt.subplots(1, 1)
axes.scatter(z_1, y_1, label="data", color="C0")
axes.scatter(z_2, y_2, color="C1")
axes.plot(zz, kh_model.eval(best_params_1, z_proj=zz), color="C0")
axes.plot(zz, kh_model.eval(best_params_2, z_proj=zz), color="C1")
axes.set_xlabel(r"$z_{\rm proj}$, mas")
axes.set_ylabel(r"$y$, mas")
axes.plot()
# axes.axis("equal")
# plt.legend()
plt.show()

# best_params_1.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))
# best_params_2.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))
# best_params_1["a"].min = 0.1
# best_params_2["a"].min = 0.1
best_params_1["lam"].min = 10
best_params_1["lam"].max = 35
# best_params_1["a"].max = 2
# best_params_2["a"].max = 2
# best_params_1["ph"].min = 0
# best_params_1["ph"].max = 2*np.pi
# best_params_2["ph"].min = 0
# best_params_2["ph"].max = 2*np.pi
# best_params_1["theta"].min = np.deg2rad(10)
# best_params_1["theta"].max = np.deg2rad(30)
# best_params_2["theta"].min = np.deg2rad(10)
# best_params_2["theta"].max = np.deg2rad(30)

emcee_result = result_1.emcee(best_params_1, steps=10000, thin=10, workers=4, burn=3000, is_weighted=False,
                              float_behavior="chi2")
# fig = corner.corner(emcee_result.flatchain[["a", "lam", "ph", "theta"]].values[::10, :], labels=["amp", "lambda", "phase", "theta"],
fig = corner.corner(emcee_result.flatchain[["a", "lam", "ph"]].values[::10, :], labels=["amp", "lambda", "phase"],
                    show_titles=True, quantiles=[0.16, 0.50, 0.84])
plt.show()