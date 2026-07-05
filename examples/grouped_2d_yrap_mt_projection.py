import numpy as np
import matplotlib.pyplot as plt

import brass


filename = "/Users/carl/Phd/baryon_stopping/default/out-1/particles_binary.bin"

pdgs = [211, -211, 321, -321, 2212, -2212]

reader = brass.BinaryReader(
    filename,
    ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"],
    skip_elastic=True,
)


# --- 2D histogram: y_rap vs mt, grouped by pdg ---

y_mt_request = brass.HistogramRequest()
y_mt_request.quantities = ["y_rap", "mt"]
y_mt_request.axes = [
    brass.RegularAxis(100, -5.0, 5.0),
    brass.RegularAxis(100, 0.0, 3.0),
]
y_mt_request.group_by = brass.HistogramGroupBy("pdg", pdgs)


# Fill grouped 2D histogram in one file pass
hists_y_mt = brass.histogram(reader, y_mt_request)

events = reader.particle_blocks_read


# --- Shared edges, centers, widths ---

first_hist = hists_y_mt[pdgs[0]]

y_edges = np.asarray(first_hist.edges[0])
mt_edges = np.asarray(first_hist.edges[1])

y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
mt_centers = 0.5 * (mt_edges[:-1] + mt_edges[1:])

y_widths = np.diff(y_edges)
mt_widths = np.diff(mt_edges)


# --- Plot projected dN/dy by PDG ---

plt.figure()

for pdg in pdgs:
    hist = hists_y_mt[pdg]
    counts_y_mt = hist.values

    counts_y = counts_y_mt.sum(axis=1)
    dndy = counts_y / (y_widths * events)

    plt.plot(y_centers, dndy, label=str(pdg))

plt.xlabel(r"$y$")
plt.ylabel(r"$1/N_{\mathrm{ev}}\, dN/dy$")
plt.title(r"Projected rapidity distribution by PDG")
plt.grid(True)
plt.legend()


# --- Plot projected dN/dmT by PDG ---

plt.figure()

for pdg in pdgs:
    hist = hists_y_mt[pdg]
    counts_y_mt = hist.values

    counts_mt = counts_y_mt.sum(axis=0)
    dndmt = counts_mt / (mt_widths * events)

    plt.plot(mt_centers, dndmt, label=str(pdg))

plt.xlabel(r"$m_T$")
plt.ylabel(r"$1/N_{\mathrm{ev}}\, dN/dm_T$")
plt.title(r"Projected transverse mass distribution by PDG")
plt.grid(True)
plt.legend()


# --- Optional: plot 2D density for each PDG ---

for pdg in pdgs:
    hist = hists_y_mt[pdg]
    counts_y_mt = hist.values

    density_y_mt = counts_y_mt / events
    density_y_mt = density_y_mt / y_widths[:, None]
    density_y_mt = density_y_mt / mt_widths[None, :]

    plt.figure()
    plt.pcolormesh(
        y_edges,
        mt_edges,
        density_y_mt.T,
        shading="auto",
    )
    plt.xlabel(r"$y$")
    plt.ylabel(r"$m_T$")
    plt.title(rf"PDG {pdg}: $1/N_{{\mathrm{{ev}}}}\, d^2N/(dy\,dm_T)$")
    plt.colorbar(label=r"$1/N_{\mathrm{ev}}\, d^2N/(dy\,dm_T)$")


plt.show()
