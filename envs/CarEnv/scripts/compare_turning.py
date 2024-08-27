import matplotlib.pyplot as plt
import numpy as np

from CarEnv.Configs import VEH_CAR_DYNAMIC, VEH_CAR_KINEMATIC
from CarEnv.Physics.BicycleModel import BicycleModel
from CarEnv.Physics.SingleTrackDugoffModel import SingleTrackDugoffModel
from CarEnv.Physics.VelocityController import LinearVelocityController, SimpleEngineDragVelocityController

conf_kin = {k: v for k, v in VEH_CAR_KINEMATIC.items() if k != 'type'}
conf_dyn = {k: v for k, v in VEH_CAR_DYNAMIC.items() if k != 'type'}

veh_kin = BicycleModel(LinearVelocityController(), **conf_kin)
veh_dyn = SingleTrackDugoffModel(**conf_dyn)

veh_kin.reset()
veh_dyn.reset()

ts = np.linspace(0, 7, 7 * 100 + 1)
dt = ts[1] - ts[0]

p_kin = []
v_kin = []
p_dyn = []
v_dyn = []
a_dyn_control = []

for t in ts:
    s_control = t > 4.

    a_control = max(0., 5. - veh_dyn.v_loc_[0]) * .3
    a_dyn_control.append(a_control)
    veh_dyn.update(np.array([s_control, a_control, 0.]), dt)
    p_dyn.append(veh_dyn.get_pose()[:2])
    v_dyn.append(veh_dyn.v_loc_[0])

    v_control = v_dyn[-1]
    v = veh_kin.update(np.array((s_control, v_control)), dt)['v_new']
    p_kin.append(veh_kin.get_pose()[:2])
    v_kin.append(v)

p_kin = np.stack(p_kin)
p_dyn = np.stack(p_dyn)

plt.figure()
plt.plot(*p_kin.T, label='kin')
plt.plot(*p_dyn.T, label='dyn')
plt.xlabel('$p_x$')
plt.ylabel('$p_y$')
plt.legend()
plt.axis('equal')

plt.figure()
plt.plot(ts, v_kin, label='kin')
plt.plot(ts, v_dyn, label='dyn')
plt.xlabel('$t$')
plt.ylabel('$v_{x,loc}$')
plt.legend()

plt.figure()
plt.plot(ts, a_dyn_control)
plt.xlabel('$t$')
plt.ylabel('$c_{\\mathrm{Eng}}$')

plt.show()
