# Doom demo asset and license notes

This repository does not contain a Doom IWAD, ViZDoom binary, scenario WAD, or
game screenshot. The separate game environment installs `vizdoom==1.3.1`, and
the demo resolves these files from that installed package at runtime:

- `freedoom2.wad`: Freedoom 0.13 game content, BSD 3-Clause. The license text is
  reproduced in `FREEDOOM-BSD-3-CLAUSE.txt`.
- `scenarios/basic.wad`: the ViZDoom `basic` research scenario distributed with
  ViZDoom 1.3.1.
- ViZDoom engine and Python bindings: ViZDoom 1.3.1. Code original to ViZDoom is
  MIT licensed; its embedded ZDoom-derived engine includes components under
  several licenses. Consult the notices shipped with the ViZDoom source and
  binary distribution for the complete component terms.

Primary sources checked:

- ViZDoom 1.3.1 release/tag (`f771231811cd3be417f97230d009e0aa9d983ed6`):
  https://github.com/Farama-Foundation/ViZDoom/releases/tag/1.3.1
- ViZDoom project and licensing statement:
  https://github.com/Farama-Foundation/ViZDoom/tree/1.3.1
- ViZDoom `basic` scenario configuration:
  https://github.com/Farama-Foundation/ViZDoom/blob/1.3.1/scenarios/basic.cfg
- Freedoom 0.13 license:
  https://github.com/freedoom/freedoom/blob/v0.13.0/COPYING.adoc
- Freedoom project description and asset licensing:
  https://freedoom.github.io/about.html

Recorded PNGs are generated during an episode from ViZDoom's screen buffer and
belong in the run output directory. They are not source assets and should not be
committed here.
