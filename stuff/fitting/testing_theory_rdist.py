import numpy as np
import matplotlib.pyplot as plt
from colicoords.models import RDistModel, PSF

num_points = 51
x_range = 5
dx = (2 * x_range + 1) / num_points

x = np.arange(dx, x_range + dx, dx)
x = np.r_[-x[::-1], 0, x]

print(x)
print(x[20:-20])
print(np.min(np.abs(x)))

psf = PSF(sigma=160/80)
rmodel = RDistModel(psf, 40)

x, y = rmodel.signal_cytosol(5.5)

plt.plot(x, y)
plt.xlim(0, 20)

plt.show()



dx = 0.02
x = np.arange(-4, 4, dx) + dx/2

rho = 1
r = 1
cs = np.sqrt(rho*2*np.sqrt(r**2 - x**2))

# plt.plot(x, cs)
# plt.xlim(0, 1000)
# plt.show()
x = np.linspace(-3, 3, num=15000)
cI = np.sqrt(rho * np.sqrt((1 + (x**2) / (r**2 - x**2))))
cI = np.nan_to_num(cI)
print(np.unique(cI))
print(cI[-100:])

plt.plot(x, cI)
plt.title('cI')
plt.xlim(-4, 4)
plt.show()

sig = r/5
print('sig', sig)
dx = np.diff(x)[0]
gx = np.arange(-3*sig, 3*sig, dx)
gaussian = np.exp(-(gx/sig)**2/2)


mu = 0
y = np.exp(-((x-mu)/sig)**2/2)*cI
integrated = np.array([np.trapz(np.exp(-((x-mu)/sig)**2/2)*cI, x=x) for mu in np.arange(-4, 4, dx)])

result = np.convolve(cI, gaussian)#, mode="full")

xarr, r = 4, 1
print('wut', np.divide(xarr**2 , (r**2 - xarr**2)))

def sim_r_dist(r, sigma):
    xmin, xmax = -4*r, 4*r
    xarr = np.linspace(xmin, xmax, num=2500)

    sqrt_arg = 1 + (xarr**2 / (r**2 - xarr**2))
    sqrt_arg[sqrt_arg < 0] = 0
    yvals = np.sqrt(sqrt_arg)

    dx = xarr[1] - xarr[0]
    gx = np.arange(-3*sigma, 3*sigma, dx)
    gy = np.exp(-((gx/sigma)**2 / 2))


    res = np.convolve(yvals, gy)

    #convoluted = np.array([np.trapz(np.exp(-((xarr - mu)/sigma)**2 / 2) * yvals, x=xarr) for mu in xarr])


    i = int((len(gx) - 1) / 2)
    return xarr, res[i:-i]

x, res = sim_r_dist(1, 0.4)

print('new len', len(x), len(res))


plt.plot(res)
plt.title('conv 1, 0.2')
plt.show()


#
# x_arr, conv = sim_r_dist(380, 380/5)
#
# plt.plot(x_arr, conv)
# plt.title('conv 380, 200')
# plt.show()
#

#conv = np.convolve(cI, psf)

#
# plt.plot(gx, gaussian)
# plt.title('gauss')
# plt.show()
#
#
# x_new = np.cumsum(np.repeat(dx, len(result)))
#
# plt.title('result')
# plt.plot(x_new, result)
# plt.show()
#
# plt.title('integrated')
# plt.plot(integrated)
# plt.show()