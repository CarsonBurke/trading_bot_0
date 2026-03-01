// Catppuccin Mocha color palette
// Used across TUI and chart generation for consistent theming

pub mod catppuccin {
    // Background colors
    pub const BASE: (u8, u8, u8) = (30, 30, 46); // #1e1e2e
    pub const MANTLE: (u8, u8, u8) = (24, 24, 37); // #181825
    pub const CRUST: (u8, u8, u8) = (17, 17, 27); // #11111b

    // Text colors
    pub const TEXT: (u8, u8, u8) = (205, 214, 244); // #cdd6f4
    pub const SUBTEXT0: (u8, u8, u8) = (166, 173, 200); // #a6adc8
    pub const SUBTEXT1: (u8, u8, u8) = (186, 194, 222); // #bac2de

    // Surface colors
    pub const SURFACE0: (u8, u8, u8) = (49, 50, 68); // #313244
    pub const SURFACE1: (u8, u8, u8) = (69, 71, 90); // #45475a
    pub const SURFACE2: (u8, u8, u8) = (88, 91, 112); // #585b70

    // Overlay colors
    pub const OVERLAY0: (u8, u8, u8) = (108, 112, 134); // #6c7086
    pub const OVERLAY1: (u8, u8, u8) = (127, 132, 156); // #7f849c
    pub const OVERLAY2: (u8, u8, u8) = (147, 153, 178); // #9399b2

    // Accent colors
    pub const BLUE: (u8, u8, u8) = (137, 180, 250); // #89b4fa
    pub const LAVENDER: (u8, u8, u8) = (180, 190, 254); // #b4befe
    pub const SAPPHIRE: (u8, u8, u8) = (116, 199, 236); // #74c7ec
    pub const SKY: (u8, u8, u8) = (137, 220, 235); // #89dceb
    pub const TEAL: (u8, u8, u8) = (148, 226, 213); // #94e2d5
    pub const GREEN: (u8, u8, u8) = (166, 227, 161); // #a6e3a1
    pub const YELLOW: (u8, u8, u8) = (249, 226, 175); // #f9e2af
    pub const PEACH: (u8, u8, u8) = (250, 179, 135); // #fab387
    pub const MAROON: (u8, u8, u8) = (235, 160, 172); // #eba0ac
    pub const RED: (u8, u8, u8) = (243, 139, 168); // #f38ba8
    pub const MAUVE: (u8, u8, u8) = (203, 166, 247); // #cba6f7
    pub const PINK: (u8, u8, u8) = (245, 194, 231); // #f5c2e7
    pub const FLAMINGO: (u8, u8, u8) = (242, 205, 205); // #f2cdcd
    pub const ROSEWATER: (u8, u8, u8) = (245, 224, 220); // #f5e0dc
}

#[cfg(feature = "ratatui")]
pub mod ratatui_colors {
    use super::catppuccin;
    use ratatui::style::Color;

    pub const BASE: Color = Color::Rgb(catppuccin::BASE.0, catppuccin::BASE.1, catppuccin::BASE.2);
    pub const MANTLE: Color = Color::Rgb(
        catppuccin::MANTLE.0,
        catppuccin::MANTLE.1,
        catppuccin::MANTLE.2,
    );
    pub const CRUST: Color = Color::Rgb(
        catppuccin::CRUST.0,
        catppuccin::CRUST.1,
        catppuccin::CRUST.2,
    );
    pub const TEXT: Color = Color::Rgb(catppuccin::TEXT.0, catppuccin::TEXT.1, catppuccin::TEXT.2);
    pub const SUBTEXT0: Color = Color::Rgb(
        catppuccin::SUBTEXT0.0,
        catppuccin::SUBTEXT0.1,
        catppuccin::SUBTEXT0.2,
    );
    pub const SUBTEXT1: Color = Color::Rgb(
        catppuccin::SUBTEXT1.0,
        catppuccin::SUBTEXT1.1,
        catppuccin::SUBTEXT1.2,
    );
    pub const SURFACE0: Color = Color::Rgb(
        catppuccin::SURFACE0.0,
        catppuccin::SURFACE0.1,
        catppuccin::SURFACE0.2,
    );
    pub const SURFACE1: Color = Color::Rgb(
        catppuccin::SURFACE1.0,
        catppuccin::SURFACE1.1,
        catppuccin::SURFACE1.2,
    );
    pub const SURFACE2: Color = Color::Rgb(
        catppuccin::SURFACE2.0,
        catppuccin::SURFACE2.1,
        catppuccin::SURFACE2.2,
    );
    pub const OVERLAY0: Color = Color::Rgb(
        catppuccin::OVERLAY0.0,
        catppuccin::OVERLAY0.1,
        catppuccin::OVERLAY0.2,
    );
    pub const OVERLAY1: Color = Color::Rgb(
        catppuccin::OVERLAY1.0,
        catppuccin::OVERLAY1.1,
        catppuccin::OVERLAY1.2,
    );
    pub const OVERLAY2: Color = Color::Rgb(
        catppuccin::OVERLAY2.0,
        catppuccin::OVERLAY2.1,
        catppuccin::OVERLAY2.2,
    );
    pub const BLUE: Color = Color::Rgb(catppuccin::BLUE.0, catppuccin::BLUE.1, catppuccin::BLUE.2);
    pub const LAVENDER: Color = Color::Rgb(
        catppuccin::LAVENDER.0,
        catppuccin::LAVENDER.1,
        catppuccin::LAVENDER.2,
    );
    pub const SAPPHIRE: Color = Color::Rgb(
        catppuccin::SAPPHIRE.0,
        catppuccin::SAPPHIRE.1,
        catppuccin::SAPPHIRE.2,
    );
    pub const SKY: Color = Color::Rgb(catppuccin::SKY.0, catppuccin::SKY.1, catppuccin::SKY.2);
    pub const TEAL: Color = Color::Rgb(catppuccin::TEAL.0, catppuccin::TEAL.1, catppuccin::TEAL.2);
    pub const GREEN: Color = Color::Rgb(
        catppuccin::GREEN.0,
        catppuccin::GREEN.1,
        catppuccin::GREEN.2,
    );
    pub const YELLOW: Color = Color::Rgb(
        catppuccin::YELLOW.0,
        catppuccin::YELLOW.1,
        catppuccin::YELLOW.2,
    );
    pub const PEACH: Color = Color::Rgb(
        catppuccin::PEACH.0,
        catppuccin::PEACH.1,
        catppuccin::PEACH.2,
    );
    pub const MAROON: Color = Color::Rgb(
        catppuccin::MAROON.0,
        catppuccin::MAROON.1,
        catppuccin::MAROON.2,
    );
    pub const RED: Color = Color::Rgb(catppuccin::RED.0, catppuccin::RED.1, catppuccin::RED.2);
    pub const MAUVE: Color = Color::Rgb(
        catppuccin::MAUVE.0,
        catppuccin::MAUVE.1,
        catppuccin::MAUVE.2,
    );
    pub const PINK: Color = Color::Rgb(catppuccin::PINK.0, catppuccin::PINK.1, catppuccin::PINK.2);
    pub const FLAMINGO: Color = Color::Rgb(
        catppuccin::FLAMINGO.0,
        catppuccin::FLAMINGO.1,
        catppuccin::FLAMINGO.2,
    );
    pub const ROSEWATER: Color = Color::Rgb(
        catppuccin::ROSEWATER.0,
        catppuccin::ROSEWATER.1,
        catppuccin::ROSEWATER.2,
    );
}

#[cfg(feature = "plotters")]
pub mod plotters_colors {
    use super::catppuccin;
    use plotters::style::RGBColor;

    pub const BASE: RGBColor = RGBColor(catppuccin::BASE.0, catppuccin::BASE.1, catppuccin::BASE.2);
    pub const MANTLE: RGBColor = RGBColor(
        catppuccin::MANTLE.0,
        catppuccin::MANTLE.1,
        catppuccin::MANTLE.2,
    );
    pub const CRUST: RGBColor = RGBColor(
        catppuccin::CRUST.0,
        catppuccin::CRUST.1,
        catppuccin::CRUST.2,
    );
    pub const TEXT: RGBColor = RGBColor(catppuccin::TEXT.0, catppuccin::TEXT.1, catppuccin::TEXT.2);
    pub const SUBTEXT0: RGBColor = RGBColor(
        catppuccin::SUBTEXT0.0,
        catppuccin::SUBTEXT0.1,
        catppuccin::SUBTEXT0.2,
    );
    pub const SUBTEXT1: RGBColor = RGBColor(
        catppuccin::SUBTEXT1.0,
        catppuccin::SUBTEXT1.1,
        catppuccin::SUBTEXT1.2,
    );
    pub const SURFACE0: RGBColor = RGBColor(
        catppuccin::SURFACE0.0,
        catppuccin::SURFACE0.1,
        catppuccin::SURFACE0.2,
    );
    pub const SURFACE1: RGBColor = RGBColor(
        catppuccin::SURFACE1.0,
        catppuccin::SURFACE1.1,
        catppuccin::SURFACE1.2,
    );
    pub const SURFACE2: RGBColor = RGBColor(
        catppuccin::SURFACE2.0,
        catppuccin::SURFACE2.1,
        catppuccin::SURFACE2.2,
    );
    pub const OVERLAY0: RGBColor = RGBColor(
        catppuccin::OVERLAY0.0,
        catppuccin::OVERLAY0.1,
        catppuccin::OVERLAY0.2,
    );
    pub const OVERLAY1: RGBColor = RGBColor(
        catppuccin::OVERLAY1.0,
        catppuccin::OVERLAY1.1,
        catppuccin::OVERLAY1.2,
    );
    pub const OVERLAY2: RGBColor = RGBColor(
        catppuccin::OVERLAY2.0,
        catppuccin::OVERLAY2.1,
        catppuccin::OVERLAY2.2,
    );
    pub const BLUE: RGBColor = RGBColor(catppuccin::BLUE.0, catppuccin::BLUE.1, catppuccin::BLUE.2);
    pub const LAVENDER: RGBColor = RGBColor(
        catppuccin::LAVENDER.0,
        catppuccin::LAVENDER.1,
        catppuccin::LAVENDER.2,
    );
    pub const SAPPHIRE: RGBColor = RGBColor(
        catppuccin::SAPPHIRE.0,
        catppuccin::SAPPHIRE.1,
        catppuccin::SAPPHIRE.2,
    );
    pub const SKY: RGBColor = RGBColor(catppuccin::SKY.0, catppuccin::SKY.1, catppuccin::SKY.2);
    pub const TEAL: RGBColor = RGBColor(catppuccin::TEAL.0, catppuccin::TEAL.1, catppuccin::TEAL.2);
    pub const GREEN: RGBColor = RGBColor(
        catppuccin::GREEN.0,
        catppuccin::GREEN.1,
        catppuccin::GREEN.2,
    );
    pub const YELLOW: RGBColor = RGBColor(
        catppuccin::YELLOW.0,
        catppuccin::YELLOW.1,
        catppuccin::YELLOW.2,
    );
    pub const PEACH: RGBColor = RGBColor(
        catppuccin::PEACH.0,
        catppuccin::PEACH.1,
        catppuccin::PEACH.2,
    );
    pub const MAROON: RGBColor = RGBColor(
        catppuccin::MAROON.0,
        catppuccin::MAROON.1,
        catppuccin::MAROON.2,
    );
    pub const RED: RGBColor = RGBColor(catppuccin::RED.0, catppuccin::RED.1, catppuccin::RED.2);
    pub const MAUVE: RGBColor = RGBColor(
        catppuccin::MAUVE.0,
        catppuccin::MAUVE.1,
        catppuccin::MAUVE.2,
    );
    pub const PINK: RGBColor = RGBColor(catppuccin::PINK.0, catppuccin::PINK.1, catppuccin::PINK.2);
    pub const FLAMINGO: RGBColor = RGBColor(
        catppuccin::FLAMINGO.0,
        catppuccin::FLAMINGO.1,
        catppuccin::FLAMINGO.2,
    );
    pub const ROSEWATER: RGBColor = RGBColor(
        catppuccin::ROSEWATER.0,
        catppuccin::ROSEWATER.1,
        catppuccin::ROSEWATER.2,
    );
}
