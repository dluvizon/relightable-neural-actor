# Borrowed from Relighting4D

import numpy as np
import torch


def safe_divide(x, denom, eps=1e-7):
    # div = torch.div(x, denom + eps)
    div = torch.div(x, torch.clamp(denom, eps, None))
    ret = div
    ret[div > 1e7] = 0.

    return ret


class Microfacet:
    """As described in:
        Microfacet Models for Refraction through Rough Surfaces [EGSR '07]
    """
    def __init__(self, default_rough=0.3, lambert_only=False, specular_only=False, f0=0.91):
        self.default_rough = default_rough
        self.lambert_only = lambert_only
        self.specular_only = specular_only
        self.f0 = f0

    def __call__(self, pts2l, pts2c, normal, albedo=None, rough=None):
        """All in the world coordinates.

        Too low roughness is OK in the forward pass, but may be numerically
        unstable in the backward pass

        pts2l: NxLx3
        pts2c: Nx3
        normal: Nx3
        albedo: Nx3
        rough: Nx1
        """
        if albedo is None:
            albedo = torch.ones(pts2c.shape[0], 3, device=pts2l.device)
        if rough is None:
            rough = self.default_rough * torch.ones(pts2c.shape[0], 1, device=pts2l.device)

        # Normalize directions and normals
        pts2l = torch.nn.functional.normalize(pts2l, p=2, dim=-1, eps=1e-7)
        pts2c = torch.nn.functional.normalize(pts2c, p=2, dim=-1, eps=1e-7)
        normal = torch.nn.functional.normalize(normal, p=2, dim=-1, eps=1e-7)

        # Glossy
        h = pts2l + pts2c[:, None, :] # NxLx3
        h = torch.nn.functional.normalize(h, p=2, dim=-1, eps=1e-7)

        f = self._get_f(pts2l, h) # NxL

        alpha = rough ** 2
        d = self._get_d(h, normal, alpha=alpha) # NxL
        g = self._get_g(pts2c, h, normal, alpha=alpha) # NxL

        l_dot_n = torch.einsum('ijk,ik->ij', pts2l, normal)
        v_dot_n = torch.einsum('ij,ij->i', pts2c, normal)
        denom = 4 * torch.abs(l_dot_n) * torch.abs(v_dot_n)[:, None]
        microfacet = safe_divide(f * g * d, denom) # NxL
        brdf_glossy = microfacet[:, :, None].repeat(1, 1, 3)

        # Diffuse
        lambert = albedo[:, None, :] / np.pi # Nx3
        brdf_diffuse = lambert
        # brdf_diffuse = tf.broadcast_to(
        #     lambert[:, None, :], tf.shape(brdf_glossy)) # NxLx3
        # Mix two shaders
        if self.lambert_only:
            brdf = brdf_diffuse
        elif self.specular_only:
            brdf = brdf_glossy
        else:
            brdf = brdf_glossy + brdf_diffuse # TODO: energy conservation?

        return brdf # NxLx3


    @staticmethod
    def _get_g(v, m, n, alpha=0.1):
        """Geometric function (GGX).

        v: Nx3
        m: NxLx3
        n: Nx3
        """
        cos_theta_v = torch.einsum('ij,ij->i', n, v)
        cos_theta = torch.einsum('ijk,ik->ij', m, v)
        denom = cos_theta_v[:, None]
        div = safe_divide(cos_theta, denom)
        chi = torch.where(div > 0, 1., 0.)

        cos_theta_v_sq = torch.square(cos_theta_v)
        cos_theta_v_sq = torch.clip(cos_theta_v_sq, 0., 1.)
        denom = cos_theta_v_sq
        tan_theta_v_sq = safe_divide(1 - cos_theta_v_sq, denom)
        tan_theta_v_sq = torch.clip(tan_theta_v_sq, 0., 1e10)

        denom = 1 + torch.sqrt(1 + alpha ** 2 * tan_theta_v_sq[:, None])
        g = safe_divide(chi * 2, denom)

        return g # (n_pts, n_lights)


    @staticmethod
    def _get_d(m, n, alpha=0.1):
        """Microfacet distribution (GGX).

        m: NxLx3
        n: Nx3
        """
        cos_theta_m = torch.einsum('ijk,ik->ij', m, n) # (N, L)
        chi = torch.where(cos_theta_m > 0, 1., 0.)
        cos_theta_m_sq = torch.square(cos_theta_m)
        denom = cos_theta_m_sq
        tan_theta_m_sq = safe_divide(1 - cos_theta_m_sq, denom)
        denom = np.pi * torch.square(cos_theta_m_sq) * torch.square(
            alpha ** 2 + tan_theta_m_sq)
        d = safe_divide(alpha ** 2 * chi, denom)

        return d # (n_pts, n_lights)


    def _get_f(self, l, m):
        """Fresnel (Schlick's approximation).

        l: NxLx3
        m: NxLx3
        """
        cos_theta = torch.einsum('ijk,ijk->ij', l, m)
        f = self.f0 + (1 - self.f0) * (1 - cos_theta) ** 5

        return f # (n_pts, n_lights)



if __name__ == '__main__':
    microfacet = Microfacet(f0=0.04, lambert_only=False, specular_only=True)
 
    pts2l = torch.from_numpy(np.array([[[-20.0, 0, 1]]], dtype=np.float32))
    pts2c = torch.from_numpy(np.array([[3, 0, 1]], dtype=np.float32))
    normal = torch.from_numpy(np.array([[0, 0, 1]], dtype=np.float32))
    albedo = torch.from_numpy(np.array([[1, 1, 1]], dtype=np.float32))
    rough = torch.from_numpy(np.array([[0.3]], dtype=np.float32))

    brdf = microfacet(pts2l, pts2c, normal, albedo=albedo, rough=rough)
    print ('done: ', brdf[0, 0])