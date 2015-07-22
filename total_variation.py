#!env python
# Total Variation denoising (1D, 2D)
# License: Public Domain
# 
# reference: Kei. K et al., "Fast Computation Method for Total Variation Minimization", Denshi-Jyoho-Tsushin-Gakkai Journal D, Vol. J93-D, No.3, pp.326-335.
# http://www.ams.giti.waseda.ac.jp/pdf-files/j93-d_3_326.pdf

import numpy as np
import matplotlib.pyplot as plt

def total_variation_denoising_1d(us, lmd, max_iter=100, return_history=False):
    u'''calculate denoising version of 1D array us'''
    us = np.array(us).ravel()
    gs = np.array(us)

    def shift(xs, h):
        return np.hstack([xs[h:], [xs[-1]]] if h > 0 else [[xs[0]], xs[:h]])

    def sgd(xs):
        return np.sign(-xs)

    def deltaJ(xs):
        return sgd(shift(xs, 1) - xs) + sgd(shift(xs, -1) - xs)

    us_history = [gs]
    w = 0.5
    for i in range(max_iter):
        us -= w*(us - gs + lmd*deltaJ(us))
        if return_history:
            us_history.append(np.array(list(us)))
        w *= 0.9

    if return_history:
        return us_history

    return us

def total_variation_denoising_2d(us, lmd, max_iter=100, return_history=False):
    u'''calculate denoising version of 2D array us'''
    us = np.array(us)
    gs = np.array(us)

    def xshift(xs, h):
        return np.hstack([xs[:, h:], xs[:, -2:-1]] if h > 0 else [xs[:, 0:1], xs[:, :h]])

    def yshift(xs, h):
        return np.vstack([xs[h:, :], xs[-2:-1, :]] if h > 0 else [xs[0:1, :], xs[:h, :]])

    def sgd(xs):
        return np.sign(-xs)

    def deltaJ(xs):
        return sgd(xshift(xs, 1) - xs) + sgd(xshift(xs, -1) - xs) + sgd(yshift(xs, 1) - xs) + sgd(yshift(xs, -1) - xs)

    us_history = [gs]
    w = 0.5
    for i in range(max_iter):
        us -= w*(us - gs + lmd*deltaJ(us))
        if return_history:
            us_history.append(np.array(list(us)))
        w *= 0.9

    if return_history:
        return us_history

    return us

def total_variation_test():
    # 1D test
    xs = np.linspace(-2, 2, 200)
    us = (np.arange(len(xs)) / 23) % 3

    def create_noisy_signal(us, noise):
        return us + np.random.randn(len(us)) * noise
    us_noisy = create_noisy_signal(us, 0.3)

    us_denoise = total_variation_denoising_1d(us_noisy, 0.8)
    us_history = total_variation_denoising_1d(us_noisy, 0.8, 100, True)

    fig, axs = plt.subplots(5, 1)
    fig.suptitle('Total Variation Denosing on 1D Array')
    axs[0].plot(xs, us, color='red', label='org', linestyle='--', linewidth=4, alpha=0.5)
    axs[0].plot(xs, us_noisy, color='blue', label='noisy', linewidth=2, alpha=0.5)
    axs[0].plot(xs, us_denoise, color='black', label='denoise', linewidth=2, alpha=0.8)
    axs[0].legend()

    for i, ax in enumerate(axs[1:]):
        it = 20*i + 1
        us_denoise = us_history[it]
        ax.plot(xs, us, color='red', label='org', linestyle='--', linewidth=1, alpha=0.5)
        ax.plot(xs, us_denoise, label='iter %d' % it, linewidth=1, alpha=0.5)
        ax.legend()

    # 2D test
    import skimage
    import skimage.data
    import skimage.transform
    import skimage.color

    img = skimage.data.lena()
    img = skimage.transform.resize(img, (128, 128))
    img_noise = img + np.random.randn(*img.shape) * 0.05
    lmd = 0.3
    img_denoise = np.dstack([total_variation_denoising_2d(img_noise[:, :, ch], lmd) for ch in range(3)])

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Total Variation Denosing on 2D Image')
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('original')
    axs[0, 1].imshow(img_noise)
    axs[0, 1].set_title('noise')

    denoise_axs = [axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]
    for ax, lmd in zip(denoise_axs, [0.05, 0.1, 0.3, 0.5]):
        ax.imshow(np.dstack([total_variation_denoising_2d(img_noise[:, :, ch], lmd) for ch in range(3)]))
        ax.set_title('denoise $\\lambda = %f$' % lmd)

if __name__=='__main__':
    total_variation_test()
    plt.show()
