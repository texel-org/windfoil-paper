# Fixtures

Image bytes are ignored by git. The tracked manifests hold each source URL, its
expected size and its SHA256, so a fixture is reproducible and verifiable
without redistributing it.

```sh
node tools/fetch-fixtures.js all          # everything declared here
node tools/fetch-fixtures.js kodak24      # or one at a time
npm run fixtures:farlev                   # farlev-highres, the demos' default target
```

Files already present are re-verified rather than re-downloaded. `--force`
replaces a file that fails verification; downloads over 1 GiB also need
`--accept-large`. Unsafe paths, non-HTTPS sources, wrong sizes and checksum
mismatches are all rejected.

The browser demo uses a 4096 px copy of the Färlev photograph,
`farlev-dip-in-road-4096.jpg`, which `npm run web:demo` and `npm run web:build`
derive from the fixture with `tools/web-image.js`.

| Manifest | Contents | Size |
| --- | --- | ---: |
| `kodak24` | The Kodak lossless set: 24 PNGs, native 768×512 (portraits 512×768) → `fixtures/kodak/` | 15.4 MB |
| `farlev-highres` | W. Carter's Färlev photograph, native 4925×2770 → `fixtures/wikimedia/` | 14.0 MB |

Sources and terms:

- Kodak — <https://r0k.us/graphics/kodak/>. The host describes the set as
  released for unrestricted use but provides no formal license; check the
  source terms before redistributing it.
- Färlev — <https://commons.wikimedia.org/wiki/File:A_dip_in_the_road_in_F%C3%A4rlev.jpg>.
  CC0 1.0, author W. Carter.
