---
title: Solarized
description: Precision colors for machines and people
author: Ethan Schoonover
tags: test, testing, test123
colors: light yellow
created:  2011 Mar 15
modified: 2011 Apr 16

---

Solarized
=========

## Random Weird Things

[![solarized dualmode](https://github.com/altercation/solarized/raw/master/img/solarized-yinyang.png)](#features)

Solarized is a sixteen color palette (eight monotones, eight accent colors)
designed for use with terminal and gui applications. It has several [unique
properties](#features). I designed this colorscheme with both precise
[CIELAB](http://en.wikipedia.org/wiki/Lab_color_space) lightness relationships
and a refined set of hues based on fixed color wheel relationships. It has been
tested extensively in real world use on color calibrated displays (as well as
uncalibrated/intentionally miscalibrated displays) and in a variety of lighting
conditions.

***See the [changelog] for what's new in the most recent release.***

![solarized palette](https://github.com/altercation/solarized/raw/master/img/solarized-palette.png)

![solarized vim](https://github.com/altercation/solarized/raw/master/img/solarized-vim.png)

Currently available in formats for (cf [screenshots](#screenshots) below):

### Editors & IDEs

*   **Vim** by [me](https://github.com/altercation) (the Vim-only portion of Solarized is
    [available here](https://github.com/altercation/vim-colors-solarized), for use with
    Pathogen, etc.). See also the [Vim README](http://ethanschoonover.com/solarized/vim-colors-solarized).
*   **Emacs** courtesy of [Greg Pfeil](http://blog.technomadic.org)
    ([@sellout](http://twitter.com/sellout))
    in the main repo and in a [standalone repository][Emacs Repository]
*   **IntelliJ IDEA**
    courtesy of [Johan Kaving](https://github.com/jkaving) and
    ([@flangy](http://twitter.com/flangy))
    in the main repo and in a [standalone repository][IntelliJ Repository]
*   **NetBeans** courtesy of [Brian Fenton](https://github.com/fentie) and
    in the main repo and in a [standalone repository][NetBeans Repository]
*   **SeeStyle theme for Coda & SubEthaEdit** courtesy of
    [Justin Hileman](http://justinhileman.com/)
    ([@bobthecow](http://twitter.com/bobthecow)),
    in the main repo and in a
    [standalone repository][SeeStyle-Coda-SubEthaEdit Repository]
*   **TextMate** --- ***NOTE:*** Dark Theme is work in progress\
    courtesy of [Tom Martin](http://thedeplorableword.net/)
    ([@deplorableword](http://twitter.com/deplorableword))
    in the main repo and in a [standalone repository][TextMate Repository]
    (with key work from [Mark Story](http://mark-story.com)
    and [Brian Mathiyakom](http://brian.rarevisions.net))
*   **TextWrangler & BBEdit** courtesy of [Rui Carmo](http://the.taoofmac.com)
    ([@taoofmac](http://twitter.com/taoofmac))
    in the main repo and in a [standalone repository][TextWrangler-BBEdit Repository]
*   **Visual Studio** courtesy of [David Thibault](http://www.leddt.com)
    ([@leddt](http://twitter.com/leddt))
    in the main repo and in a [standalone repository][Visual Studio Repository]

*   **Xcode** work in progress ports are available for [Xcode 3] and [Xcode 4]
    and will be pulled into the main Solarized project soon.

### Terminal Emulators

* **Xresources** / Xdefaults
* **iTerm2**
* **OS X Terminal.app**
* **Putty** courtesy [Brant Bobby](http://www.control-v.net)
    and on [GitHub](https://github.com/brantb)
* **Xfce terminal** courtesy [Sasha Gerrand](http://sgerrand.com)
    and on [GitHub](https://github.com/sgerrand)

### Other Applications

*   **Mutt** e-mail client also by [me] (*just* the Mutt colorscheme is
    [available here][Mutt Repository])

### Palettes

* **Adobe Photoshop** Palette (inc. L\*a\*b values)
* **Apple Color Picker** Palettes
* **GIMP** Palette

Don't see the application you want to use it in? Download the palettes (or pull
the values from the table below) and create your own. Submit it back and I'll
happily note the contribution and include it on this page.  See also the
[Usage & Development](#usage-development) section below for details on the
specific values to be used in different contexts.

Download
--------

### [Click here to download latest version](http://ethanschoonover.com/solarized/files/solarized.zip)

Current release is **v1.0.0beta2**. See the [changelog] for details on what's
new in this release.

### Fresh Code on GitHub

You can also use the following links to access application specific downloads
and git repositories:

*   **Canonical Project Page:**

    Downloads, screenshots and more information are always available from the
    project page: <http://ethanschoonover.com/solarized>

*   **Full Git Repository:**

    The full git repository is at: <https://github.com/altercation/solarized>
    Get it using the following command:

        $ git clone git://github.com/altercation/solarized.git

*   **Application Specific Repositories:**

    You can clone repositories specific to many of the application specific
    color themes. See links in the list above or select from this list:

    * [Vim Repository]
    * [Mutt Repository]
    * [Emacs Repository]
    * [IntelliJ Repository]
    * [NetBeans Repository]
    * [SeeStyle-Coda-SubEthaEdit Repository]
    * [TextMate Repository]
    * [TextWrangler-BBEdit Repository]
    * [Visual Studio Repository]

    * [Xcode 3 work in progress][Xcode 3]
    * [Xcode 4 work in progress][Xcode 4]

Note that through the magic of [git-subtree](https://github.com/apenwarr/git-subtree)
these repositories are all kept in sync, so you can pull any of them and get the most up-to-date version.

Features
--------

1. **Selective contrast**

    On a sunny summer day I love to read a book outside. Not right in the sun;
    that's too bright. I'll hunt for a shady spot under a tree. The shaded
    paper contrasts with the crisp text nicely. If you were to actually measure
    the contrast between the two, you'd find it is much lower than black text
    on a white background (or white on black) on your display device of choice.
    Black text on white from a computer display is akin to reading a book in
    direct sunlight and tires the eye.

    ![solarized selective contrast](https://github.com/altercation/solarized/raw/master/img/solarized-selcon.png)

    Solarized reduces *brightness contrast* but, unlike many low contrast
    colorschemes, retains *contrasting hues* (based on colorwheel relations)
    for syntax highlighting readability.

2. **Both sides of the force**

    ![solarized dualmode](https://github.com/altercation/solarized/raw/master/img/solarized-dualmode.png)

    I often switch between dark and light modes when editing text and code.
    Solarized retains the same selective contrast relationships and overall
    feel when switching between the light and dark background modes. A *lot* of
    thought, planning and testing has gone into making both modes feel like
    part of a unified colorscheme.

3. **16/5 palette modes**

    ![solarized palettes](https://github.com/altercation/solarized/raw/master/img/solarized-165.png)

    Solarized works as a sixteen color palette for compatibility with common
    terminal based applications / emulators. In addition, it has been carefully
    designed to scale down to a variety of five color palettes (four base
    monotones plus one accent color) for use in design work such as web design.
    In every case it retains a strong personality but doesn't overwhelm.

5.  **Precision, symmetry**

    ![solarized symmetry](https://github.com/altercation/solarized/raw/master/img/solarized-sym.png)

    The monotones have symmetric CIELAB lightness differences, so switching
    from dark to light mode retains the same perceived contrast in brightness
    between each value. Each mode is equally readable. The accent colors are
    based off specific colorwheel relations and subsequently translated to
    CIELAB to ensure perceptual uniformity in terms of lightness. The hues
    themselves, as with the monotone \*a\*b values, have been adjusted within
    a small range to achieve the most pleasing combination of colors.

    See also the [Usage & Development](#usage-development) section below for
    details on the specific values to be used in different contexts.

    This makes colorscheme inversion trivial. Here, for instance, is a sass
    (scss) snippet that inverts solarized based on the class of the html tag
    (e.g. `<html class="dark red">` to give a dark background with red accent):

        $base03:    #002b36;
        $base02:    #073642;
        $base01:    #586e75;
        $base00:    #657b83;
        $base0:     #839496;
        $base1:     #93a1a1;
        $base2:     #eee8d5;
        $base3:     #fdf6e3;
        $yellow:    #b58900;
        $orange:    #cb4b16;
        $red:       #dc322f;
        $magenta:   #d33682;
        $violet:    #6c71c4;
        $blue:      #268bd2;
        $cyan:      #2aa198;
        $green:     #859900;
        @mixin rebase($rebase03,$rebase02,$rebase01,$rebase00,$rebase0,$rebase1,$rebase2,$rebase3)
        {
            background-color:$rebase03;
            color:$rebase0;
            * { color:$rebase0; }
            h1,h2,h3,h4,h5,h6 { color:$rebase1; border-color: $rebase0; }
            a, a:active, a:visited { color: $rebase1; }
        }
        @mixin accentize($accent) {
            a, a:active, a:visited, code.url { color: $accent; }
            h1,h2,h3,h4,h5,h6 {color:$accent}
        }
        /* light is default mode, so pair with general html definition */
        html, .light { @include rebase($base3,$base2,$base1,$base0,$base00,$base01,$base02,$base03)}
        .dark  { @include rebase($base03,$base02,$base01,$base00,$base0,$base1,$base2,$base3)}
        html * {
            color-profile: sRGB;
            rendering-intent: auto;
        }

    See also [the full css stylesheet for this site](https://github.com/altercation/ethanschoonover.com/blob/master/resources/css/style.css).

Installation
------------

Installation instructions for each version of the colorscheme are included in
the subdirectory README files. Note that for Vim (and possibly for Mutt) you
may want to clone the specific repository (for instance if you are using
Pathogen). See the links at the top of this file.

Font Samples
------------

Solarized has been designed to handle fonts of various weights and retain
readability, from the classic Terminus to the beefy Menlo.

![font samples - light](https://github.com/altercation/solarized/raw/master/img/solarized-fontsamples-light.png)
![font samples - dark](https://github.com/altercation/solarized/raw/master/img/solarized-fontsamples-dark.png)

Clockwise from upper left: Menlo, Letter Gothic, Terminus, Andale Mono.

Preview all code samples in specific font faces by selecting a link from this
list:

* [DejaVu Sans 18](http://ethanschoonover.com/solarized/img/dejavusans18/)
* [DejaVu Sans 14](http://ethanschoonover.com/solarized/img/dejavusans14/)
* [Letter Gothic 18](http://ethanschoonover.com/solarized/img/lettergothic18/)
* [Letter Gothic 14](http://ethanschoonover.com/solarized/img/lettergothic14/)

* [Andale Mono 14](http://ethanschoonover.com/solarized/img/andalemono14/)
* [Monaco 14](http://ethanschoonover.com/solarized/img/monaco14/)
* [Skyhook Mono 14](http://ethanschoonover.com/solarized/img/skyhookmono14/)

* [Terminus 12](http://ethanschoonover.com/solarized/img/terminus12/)
* [Terminus 20](http://ethanschoonover.com/solarized/img/terminus20/)

Screenshots
-----------

Click to view.

### Mutt

[![mutt dark](https://github.com/altercation/solarized/raw/master/img/screen-mutt-dark-th.png)](https://github.com/altercation/solarized/raw/master/img/screen-mutt-dark.png)
[![mutt light](https://github.com/altercation/solarized/raw/master/img/screen-mutt-light-th.png)](https://github.com/altercation/solarized/raw/master/img/screen-mutt-light.png)
