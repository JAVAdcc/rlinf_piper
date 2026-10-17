Piper gravity model provenance
==============================

``piper_no_gripper.urdf`` supplies the inertial model for Piper leader gravity compensation. It is copied unchanged from Evo-RL's ``src/lerobot/assets/piper_description/urdf/piper_no_gripper_description.urdf`` in the workstation checkout reviewed on 2026-10-17 (HEAD ``ee5c6b5830710e298c8f29ba60910dc1039be720``), under that repository's Apache-2.0 license. The source checkout also has local modifications; the content hash identifies this asset exactly.

SHA-256: ``a93c1e3fc245d9d336a427fcc506cb1fc15615268053392b249816f90c011e50``.

The driver builds the dynamics model without loading visual or collision meshes. The accompanying gravity implementation preserves the source copyright header and payload, mounting orientation, torque scaling and MIT control conventions. CPU parity against the original Pinocchio 3.9 implementation was checked with the port's Pinocchio 2.7 environment; runtime parameters are in ``PiperLeaderConfig`` and the Piper hardware YAML.
