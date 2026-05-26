import tkinter as tk
import itertools
import re
import json
from pathlib import Path
from functools import lru_cache


# =========================
# GLOBÁLNE DÁTA A NASTAVENIA
# =========================

g1_data = {}
g2_data = {}

g1_inputs = {}
g2_inputs = {}
common_inputs = {}

expanded_rule_sections = set()
last_result_payload = None
active_rule_widget = None

help_popup = None
error_popup = None
history_popup = None
history_text_output = None

RESULTS_JSON_FILE = Path("vysledky_ekvivalencie.json")


# =========================
# GUI NASTAVENIA
# =========================

BG_COLOR = '#d0e7f9'
TEXT_COLOR = '#00274d'
BUTTON_BG = '#00509e'
BUTTON_FG = 'white'

TITLE_FONT = ("Arial", 20, "bold")
LABEL_FONT = ("Arial", 14)
ENTRY_FONT = ("Arial", 14)
BUTTON_FONT = ("Arial", 16)

LHS_NONTERMINAL_PATTERN = re.compile(r"^[A-Z0-9]+'*$|^[A-Z0-9][A-Z0-9]+'*$")


# =========================
# VŠEOBECNÉ GUI FUNKCIE
# =========================

def show_frame(frame):
    frame.tkraise()


def center_popup(popup, parent):
    popup.update_idletasks()
    w = popup.winfo_width()
    h = popup.winfo_height()
    x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (w // 2)
    y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (h // 2)
    popup.geometry(f"{w}x{h}+{x}+{y}")


def create_popup(title, modal=False, resizable=False):
    popup = tk.Toplevel(root)
    popup.title(title)
    popup.configure(bg=BG_COLOR)
    popup.resizable(resizable, resizable)
    popup.transient(root)

    if modal:
        popup.grab_set()

    return popup


def create_button(parent, text, command, width=10, height=1, font=BUTTON_FONT):
    return tk.Button(
        parent,
        text=text,
        command=command,
        font=font,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        width=width,
        height=height
    )


def _clear_widget(widget):
    if widget is None:
        return

    if isinstance(widget, tk.Entry):
        widget.delete(0, tk.END)
    elif isinstance(widget, tk.Text):
        widget.delete("1.0", tk.END)


def reset_all_user_inputs():
    global last_result_payload, active_rule_widget

    g1_data.clear()
    g2_data.clear()
    expanded_rule_sections.clear()
    last_result_payload = None
    active_rule_widget = None

    for widget in g1_inputs.values():
        _clear_widget(widget)
    for widget in g2_inputs.values():
        _clear_widget(widget)
    for widget in common_inputs.values():
        _clear_widget(widget)

    if "result_content_frame" in globals():
        try:
            for child in result_content_frame.winfo_children():
                child.destroy()
            result_update_scrollbar()
        except Exception:
            pass

    if "result_text_output" in globals():
        try:
            result_text_output.config(state="normal")
            result_text_output.delete("1.0", tk.END)
            result_text_output.config(state="disabled")
            result_update_scrollbar()
        except Exception:
            pass


def close_popup_reference(popup_name):
    popup = globals().get(popup_name)

    if popup is not None:
        try:
            if popup.winfo_exists():
                popup.destroy()
        except Exception:
            pass

    globals()[popup_name] = None


# =========================
# VALIDÁCIA A PARSOVANIE VSTUPOV
# =========================

def normalize_start_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def normalize_rule_arrow(text: str) -> str:
    return (text or "").replace("→", "->")


def validate_and_parse_eq_length(eq_text: str):
    value = (eq_text or "").strip()

    if not value:
        return 0, "• L_test nemôže zostať prázdne."

    if re.fullmatch(r"\d+", value):
        return int(value), None

    return 0, "• L_test musí byť celé číslo (iba číslice 0–9)."


def collect_rule_syntax_errors(rules_lines):
    errors = []

    for idx, line in enumerate(rules_lines, start=1):
        raw_original = line.strip()
        if not raw_original:
            continue

        raw = normalize_rule_arrow(raw_original)

        if "->" not in raw:
            errors.append((idx, "Chýba '->' alebo '→'."))
            continue

        left, right = raw.split("->", 1)
        left = left.strip().upper()
        right = right.strip()

        if not left:
            errors.append((idx, "Chýba ľavá strana pred šípkou."))
        elif not LHS_NONTERMINAL_PATTERN.match(left):
            errors.append(
                (
                    idx,
                    "Ľavá strana musí byť neterminál z VEĽKÝCH písmen a/alebo číslic "
                    "(povolené aj S', A1', 12')."
                )
            )

        if right == "":
            errors.append((idx, "Chýba pravá strana za šípkou."))
        else:
            alternatives = [a.strip() for a in right.split("|")]
            for alternative in alternatives:
                if alternative == "":
                    errors.append((idx, "Prázdna alternatíva za '|'. Pre epsilon použi '()'."))

    return errors


def validate_grammar_input(label, start_raw, rules_lines):
    errors = []

    if not start_raw and not rules_lines:
        errors.append(f"• Gramatika {label} nie je úplne zadaná (chýba počiatočný symbol aj pravidlá).")
    elif not start_raw:
        errors.append(f"• Gramatika {label} nie je úplne zadaná (chýba počiatočný symbol).")
    elif not rules_lines:
        errors.append(f"• Gramatika {label} nie je úplne zadaná (chýbajú pravidlá).")

    if rules_lines:
        syntax_errors = collect_rule_syntax_errors(rules_lines)
        if syntax_errors:
            preview = "\n".join([f"    Riadok {i}: {reason}" for i, reason in syntax_errors[:8]])
            more = "" if len(syntax_errors) <= 8 else f"\n    ... a ďalších {len(syntax_errors) - 8} chýb."
            errors.append(f"• Pravidlá v {label} sú nesprávne zadané:\n" + preview + more)

    return errors


def validate_all_inputs_and_collect_errors(start1_raw, rules1_lines, start2_raw, rules2_lines):
    g1_errors = validate_grammar_input("G1", start1_raw, rules1_lines)
    g2_errors = validate_grammar_input("G2", start2_raw, rules2_lines)

    errors = []

    if g1_errors:
        errors.extend(g1_errors)

    if g1_errors and g2_errors:
        errors.append("")

    if g2_errors:
        errors.extend(g2_errors)

    return errors


# =========================
# POPUP OKNÁ
# =========================

def show_error_popup(title: str, message: str):
    global error_popup

    close_popup_reference("error_popup")

    error_popup = create_popup(title, modal=False, resizable=False)
    popup = error_popup

    body = tk.Frame(popup, bg=BG_COLOR, padx=18, pady=14)
    body.pack(fill="both", expand=True)

    tk.Label(
        body,
        text=title,
        font=("Arial", 16, "bold"),
        bg=BG_COLOR,
        fg="#b00020"
    ).pack(anchor="w")

    tk.Label(
        body,
        text=message,
        font=ENTRY_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        justify="left",
        wraplength=760
    ).pack(anchor="w", pady=(10, 15))

    def close_popup():
        close_popup_reference("error_popup")

    create_button(body, "OK", close_popup, width=10).pack(anchor="e")

    popup.bind("<Return>", lambda e: close_popup())
    popup.bind("<Escape>", lambda e: close_popup())
    popup.protocol("WM_DELETE_WINDOW", close_popup)

    center_popup(popup, root)
    popup.lift()
    popup.focus_force()


def show_text_popup(title, heading, text, modal=False, existing_popup_name=None):
    if existing_popup_name:
        existing_popup = globals().get(existing_popup_name)
        if existing_popup is not None:
            try:
                if existing_popup.winfo_exists():
                    existing_popup.lift()
                    return existing_popup
            except Exception:
                pass

    popup = create_popup(title, modal=modal, resizable=False)

    if existing_popup_name:
        globals()[existing_popup_name] = popup

    body = tk.Frame(popup, bg=BG_COLOR, padx=18, pady=14)
    body.pack(fill="both", expand=True)

    tk.Label(
        body,
        text=heading,
        font=("Arial", 16, "bold"),
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).pack(anchor="w")

    tk.Label(
        body,
        text=text,
        font=ENTRY_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        justify="left",
        wraplength=760
    ).pack(anchor="w", pady=(10, 15))

    def close_popup():
        if existing_popup_name:
            close_popup_reference(existing_popup_name)
        else:
            popup.destroy()

    btn = create_button(body, "OK", close_popup, width=10)
    btn.pack(anchor="e")

    popup.bind("<Return>", lambda e: close_popup())
    popup.bind("<Escape>", lambda e: close_popup())
    popup.protocol("WM_DELETE_WINDOW", close_popup)

    center_popup(popup, root)
    popup.focus_set()
    btn.focus_set()

    if modal:
        popup.wait_window()

    return popup


def show_intro_popup():
    intro_text = (
        "Táto aplikácia slúži na testovanie ekvivalencie dvoch "
        "bezkontextových gramatík.\n\n"
        "Po stlačení tlačidla Start sa zobrazia dve textové polia "
        "pre gramatiky G1 a G2, kde sa zadáva:\n"
        "• počiatočný symbol\n"
        "• pravidlá gramatiky\n"
        "Na to aby zbehlo otestovanie ekvivalencie je potrebné aby obe gramatiky mali "
        "zadaný užívateľom počiatočný symbol a pravidlá\n"
        "Ak ostane počiatný symbol a pravidlá prázdne stačí zadať L_test(dĺžku) "
        "a ekvivalencia sa otestuje.\n"
        "Ak je zadaný počiatočný symbol a pravidlá ostanú prázdne alebo opačne, "
        "testovanie ekvivalencie sa nevykoná\n\n"
        "Následne sa zadáva hodnota L_test, ktorá určuje maximálnu "
        "dĺžku reťazcov, do ktorej sa kontroluje ekvivalencia oboch jazykov.\n\n"
    )

    show_text_popup(
        title="Info",
        heading="Info o aplikácii",
        text=intro_text,
        modal=True
    )


def show_input_help_popup():
    help_text = (
        "1) Počiatočný symbol (S)\n"
        "   • Nie je case sensitive.\n"
        "   • To znamená, že napr. s aj S sa budú brať rovnako.\n"
        "   • Môže obsahovať veľké písmená aj číslice.\n"
        "   • Povolené sú aj apostrofy na konci, napr. S', A1'', 12'.\n\n"
        "2) Pravidlá (P -)\n"
        "   • Každý riadok zadávaj samostatne.\n"
        "   • Môžeš použiť zápis so šípkou -> aj →.\n"
        "   • Príklad:\n"
        "       S->aA | b\n"
        "       A→a | ()\n"
        "   • Alternatívy oddeľuj znakom |.\n"
        "   • Prázdne slovo zapisuj ako ().\n"
        "   • Tlačidlá nad L_test slúžia na rýchle vkladanie symbolov →, ->, | a ().\n"
        "   • Po stlačení tlačidla sa vybraný znak vloží na aktuálnu pozíciu kurzora\n\n "
        "3) L_test\n"
        "   • Určuje maximálnu dĺžku reťazcov, do ktorej sa porovnávajú jazyky.\n"
        "   • Nemôže zostať prázdne.\n"
        "   • Musí byť zadané ako celé číslo, napr. 5, 10, 12.\n"
    )

    show_text_popup(
        title="Info",
        heading="Info k zadávaniu gramatík",
        text=help_text,
        modal=False,
        existing_popup_name="help_popup"
    )


# =========================
# HISTÓRIA TESTOVANIA
# =========================

def load_history_from_json():
    if not RESULTS_JSON_FILE.exists():
        return []

    try:
        with open(RESULTS_JSON_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []
    except Exception:
        return []


def write_history_to_json(history):
    with open(RESULTS_JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=4)


def save_result_to_json(payload):
    counterexample_data = None

    if not payload.get("equivalent", False):
        counterexample = payload.get("counterexample")

        if counterexample is not None:
            word, belongs_to, not_belongs_to = counterexample
            counterexample_data = {
                "word": word,
                "display_word": "ε" if word == "" else word,
                "belongs_to": f"G{belongs_to}",
                "not_belongs_to": f"G{not_belongs_to}"
            }

    record = {
        "g1": {
            "start_symbol": g1_data.get("start", ""),
            "rules": g1_data.get("rules_lines", [])
        },
        "g2": {
            "start_symbol": g2_data.get("start", ""),
            "rules": g2_data.get("rules_lines", [])
        },
        "l_test": payload.get("eq_length"),
        "equivalent": payload.get("equivalent", False),
        "counterexample": counterexample_data
    }

    history = load_history_from_json()
    history.append(record)
    write_history_to_json(history)

    render_history_text()


def configure_history_tags(text_widget):
    text_widget.tag_configure("card_title", font=("Arial", 15, "bold"), foreground=TEXT_COLOR)
    text_widget.tag_configure("section_title", font=("Arial", 14, "bold"), foreground=TEXT_COLOR)
    text_widget.tag_configure("success", font=("Arial", 14), foreground=TEXT_COLOR)
    text_widget.tag_configure("error", font=("Arial", 14), foreground=TEXT_COLOR)
    text_widget.tag_configure("meta", font=("Arial", 12), foreground="#4a4a4a")
    text_widget.tag_configure("rule", font=("Consolas", 12), foreground=TEXT_COLOR)
    text_widget.tag_configure("separator", foreground="#6f8fa8")


def insert_history_grammar(text_widget, title, grammar):
    start_symbol = grammar.get("start_symbol", "")
    rules = grammar.get("rules", [])

    text_widget.insert(tk.END, title + "\n", ("section_title",))
    text_widget.insert(
        tk.END,
        f"Počiatočný symbol: {start_symbol if start_symbol else '(nezadaný)'}\n",
        ("meta",)
    )
    text_widget.insert(tk.END, "Pravidlá:\n", ("meta",))

    if rules:
        for rule in rules:
            text_widget.insert(tk.END, f"  {rule}\n", ("rule",))
    else:
        text_widget.insert(tk.END, "  Žiadne pravidlá\n", ("rule",))

    text_widget.insert(tk.END, "\n")


def insert_history_result(text_widget, record):
    equivalent = record.get("equivalent", False)
    counterexample = record.get("counterexample")

    if equivalent:
        text_widget.insert(
            tk.END,
            "Výsledok: Gramatiky G1 a G2 sú ekvivalentné.\n",
            ("success",)
        )
        return

    text_widget.insert(
        tk.END,
        "Výsledok: Gramatiky G1 a G2 NIE sú ekvivalentné.\n",
        ("error",)
    )

    if counterexample:
        word = counterexample.get("display_word", counterexample.get("word", ""))
        belongs_to = counterexample.get("belongs_to", "")
        not_belongs_to = counterexample.get("not_belongs_to", "")
        text_widget.insert(
            tk.END,
            f"Protipríklad: slovo {word} patrí do {belongs_to}, ale nepatrí do {not_belongs_to}.\n",
            ("error",)
        )
    else:
        text_widget.insert(tk.END, "Protipríklad: nezistený.\n", ("error",))


def render_history_text():
    global history_text_output

    if history_text_output is None:
        return

    try:
        if not history_text_output.winfo_exists():
            return

        history = load_history_from_json()

        history_text_output.config(state="normal")
        history_text_output.delete("1.0", tk.END)
        configure_history_tags(history_text_output)

        if not history:
            history_text_output.insert(
                tk.END,
                "Zatiaľ neexistuje žiadna história testovania.\n\n"
                "Po otestovaní ekvivalencie sa sem automaticky načítajú záznamy "
                "zo súboru vysledky_ekvivalencie.json."
            )
            history_text_output.config(state="disabled")
            return

        history_text_output.insert(
            tk.END,
            f"Počet uložených testovaní: {len(history)}\n",
            ("meta",)
        )
        history_text_output.insert(
            tk.END,
            "Najnovšie testovanie je zobrazené hore.\n\n",
            ("meta",)
        )

        reversed_history = list(reversed(history))

        for display_index, record in enumerate(reversed_history, start=1):
            original_index = len(history) - display_index + 1

            history_text_output.insert(tk.END, "═" * 90 + "\n", ("separator",))
            history_text_output.insert(tk.END, f"Testovanie č. {original_index}\n", ("card_title",))

            insert_history_result(history_text_output, record)

            history_text_output.insert(
                tk.END,
                f"Testované do dĺžky: {record.get('l_test', '')}\n\n",
                ("meta",)
            )

            insert_history_grammar(history_text_output, "Gramatika G1", record.get("g1", {}))
            insert_history_grammar(history_text_output, "Gramatika G2", record.get("g2", {}))
            history_text_output.insert(tk.END, "\n")

        history_text_output.config(state="disabled")
    except Exception:
        pass


def show_history_popup():
    global history_popup, history_text_output

    if history_popup is not None:
        try:
            if history_popup.winfo_exists():
                render_history_text()
                history_popup.lift()
                return
        except Exception:
            pass

    history_popup = create_popup("História testovania", modal=False, resizable=True)
    history_popup.geometry("950x600")
    history_popup.minsize(760, 450)

    history_popup.grid_rowconfigure(1, weight=1)
    history_popup.grid_columnconfigure(0, weight=1)

    title_row = tk.Frame(history_popup, bg=BG_COLOR)
    title_row.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 8))
    title_row.grid_columnconfigure(0, weight=1)

    tk.Label(
        title_row,
        text="História testovaných gramatík",
        font=("Arial", 16, "bold"),
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=0, column=0, sticky="w")

    body = tk.Frame(history_popup, bg=BG_COLOR)
    body.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 10))
    body.grid_rowconfigure(0, weight=1)
    body.grid_columnconfigure(0, weight=1)

    history_text_output = tk.Text(
        body,
        font=ENTRY_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        wrap="word",
        borderwidth=0,
        highlightthickness=0,
        state="disabled"
    )
    history_text_output.grid(row=0, column=0, sticky="nsew")

    scrollbar = tk.Scrollbar(body, orient="vertical", command=history_text_output.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    history_text_output.configure(yscrollcommand=scrollbar.set)

    btn_row = tk.Frame(history_popup, bg=BG_COLOR)
    btn_row.grid(row=2, column=0, sticky="e", padx=18, pady=(0, 14))

    create_button(btn_row, "Obnoviť", render_history_text, width=10).grid(row=0, column=0, padx=(0, 10))

    def close_history_popup():
        global history_popup, history_text_output

        try:
            if history_popup is not None and history_popup.winfo_exists():
                history_popup.destroy()
        except Exception:
            pass

        history_popup = None
        history_text_output = None

    create_button(btn_row, "Zavrieť", close_history_popup, width=10).grid(row=0, column=1)

    history_popup.protocol("WM_DELETE_WINDOW", close_history_popup)
    history_popup.bind("<Escape>", lambda e: close_history_popup())

    render_history_text()
    history_popup.lift()


# =========================
# POMOCNÉ FUNKCIE PRE PRAVIDLÁ
# =========================

def set_active_rule_widget(widget):
    global active_rule_widget
    active_rule_widget = widget


def get_active_rule_widget():
    global active_rule_widget

    focused = root.focus_get()

    if focused in (g1_inputs.get("rules"), g2_inputs.get("rules")):
        active_rule_widget = focused
        return focused

    if active_rule_widget is not None:
        try:
            if active_rule_widget.winfo_exists():
                return active_rule_widget
        except Exception:
            pass

    return g1_inputs.get("rules")


def insert_rule_symbol(symbol):
    widget = get_active_rule_widget()
    if widget is None:
        return

    widget.focus_set()
    widget.insert(tk.INSERT, symbol)
    widget.see(tk.INSERT)


def sorted_nonterminals(nonterminals):
    return sorted(nonterminals, key=len, reverse=True)


def tokenize_by_nonterminals(prod, nonterminals):
    nts = sorted_nonterminals(nonterminals)
    tokens = []
    i = 0

    while i < len(prod):
        matched = None

        for nt in nts:
            if prod.startswith(nt, i):
                matched = nt
                break

        if matched is not None:
            tokens.append(matched)
            i += len(matched)
        else:
            tokens.append(prod[i])
            i += 1

    return tokens


def join_tokens(tokens):
    return "".join(tokens)


def replace_nonterminal_token(prod, old_nt, new_nt, nonterminals):
    tokens = tokenize_by_nonterminals(prod, set(nonterminals) | {old_nt, new_nt})
    return join_tokens([new_nt if tok == old_nt else tok for tok in tokens])


def unique_preserve_order(items):
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


# =========================
# PRÁCA S GRAMATIKOU
# =========================

def process_rules(rules_input):
    rules = {}

    for rule in rules_input:
        normalized_rule = normalize_rule_arrow(rule)

        if "->" in normalized_rule:
            left, right = normalized_rule.split("->", 1)
            left = left.strip().upper()
            right_parts = [r.strip() for r in right.split("|")]
            parsed_right = ["" if r == "()" else r for r in right_parts]

            rules.setdefault(left, [])
            rules[left].extend(parsed_right)

    for left in list(rules.keys()):
        rules[left] = unique_preserve_order(rules[left])

    return rules


def find_simple_rules(grammar):
    simple_rules = {}
    nonterminals = set(grammar.keys())

    for left, productions in grammar.items():
        for prod in productions:
            tokens = tokenize_by_nonterminals(prod, nonterminals)
            if len(tokens) == 1 and tokens[0] in nonterminals:
                simple_rules.setdefault(left, []).append(tokens[0])

    return simple_rules


def remove_simple_rules(grammar, simple_rules):
    new_grammar = {key: set(value) for key, value in grammar.items()}
    changed = True

    while changed:
        changed = False
        current_simple = find_simple_rules({k: list(v) for k, v in new_grammar.items()})

        for A, B_list in current_simple.items():
            for B in B_list:
                if B in new_grammar:
                    for prod in new_grammar[B]:
                        if prod not in new_grammar[A]:
                            new_grammar[A].add(prod)
                            changed = True

    nonterminals = set(new_grammar.keys())

    for A in list(new_grammar.keys()):
        cleaned = set()

        for prod in new_grammar[A]:
            tokens = tokenize_by_nonterminals(prod, nonterminals)
            if not (len(tokens) == 1 and tokens[0] in nonterminals):
                cleaned.add(prod)

        new_grammar[A] = cleaned

    return {A: list(v) for A, v in new_grammar.items()}


def canonical_form(prod, nonterminals):
    tokens = tokenize_by_nonterminals(prod, nonterminals)
    return "".join("N" if tok in nonterminals else tok for tok in tokens)


def merge_equivalent_non_terminals_once(grammar, original_nonterminals):
    reverse_grammar = {}
    nonterminals = set(grammar.keys())

    for nt, productions in grammar.items():
        canon_prods = sorted(canonical_form(prod, nonterminals) for prod in productions)
        key = tuple(canon_prods)
        reverse_grammar.setdefault(key, []).append(nt)

    merged_grammar = dict(grammar)
    changed = False

    for nts in reverse_grammar.values():
        if len(nts) <= 1:
            continue

        candidates = [nt for nt in nts if nt in original_nonterminals]
        winner = candidates[0] if candidates else nts[0]

        for nt in nts:
            if nt == winner:
                continue

            if nt in merged_grammar:
                del merged_grammar[nt]

            current_nonterminals = set(grammar.keys())

            for A in list(merged_grammar.keys()):
                merged_grammar[A] = [
                    replace_nonterminal_token(prod, nt, winner, current_nonterminals)
                    for prod in merged_grammar[A]
                ]

            changed = True

    return merged_grammar, changed


def merge_equivalent_non_terminals_fixpoint(grammar, original_nonterminals):
    changed = True
    current = grammar

    while changed:
        current, changed = merge_equivalent_non_terminals_once(current, original_nonterminals)

    return current


def find_epsilon_producing(grammar, non_terminals):
    epsilon_nt = set()
    current_nonterminals = set(grammar.keys()) | set(non_terminals)
    changed = True

    while changed:
        changed = False

        for nt, productions in grammar.items():
            if nt in epsilon_nt:
                continue

            for prod in productions:
                if prod == "":
                    epsilon_nt.add(nt)
                    changed = True
                    break

                tokens = tokenize_by_nonterminals(prod, current_nonterminals)

                if tokens and all(tok in current_nonterminals and tok in epsilon_nt for tok in tokens):
                    epsilon_nt.add(nt)
                    changed = True
                    break

    return epsilon_nt


def remove_epsilon_productions(grammar, start_symbol, epsilon_nt):
    nonterminals = set(grammar.keys())
    new_grammar = {left: set() for left in grammar.keys()}

    for left, productions in grammar.items():
        for prod in productions:
            symbols = tokenize_by_nonterminals(prod, nonterminals)
            nullable_positions = [i for i, sym in enumerate(symbols) if sym in epsilon_nt]

            subsets = itertools.chain.from_iterable(
                itertools.combinations(nullable_positions, r)
                for r in range(len(nullable_positions) + 1)
            )

            for subset in subsets:
                new_symbols = list(symbols)

                for idx in sorted(subset, reverse=True):
                    new_symbols.pop(idx)

                new_grammar[left].add(join_tokens(new_symbols))

    for left in list(new_grammar.keys()):
        new_grammar[left].discard("")

    final_grammar = {}

    for left, prod_set in new_grammar.items():
        if prod_set:
            final_grammar[left] = list(prod_set)

    return final_grammar


def remove_productions_with_missing_nonterminals(grammar, all_nonterminals):
    """
    Odstráni pravidlá, ktoré po predchádzajúcich úpravách obsahujú
    pôvodný neterminál, ktorý už nemá v gramatike žiadne produkcie.

    Bez tejto kontroly by napríklad po odstránení pravidla B -> ε
    mohla v gramatike zostať produkcia A -> aBc. Keďže B by už nebolo
    kľúčom slovníka gramatiky, generátor by ho neskôr považoval za terminál.
    """
    current_nonterminals = set(grammar.keys())
    all_nonterminals = set(all_nonterminals)
    clean = {}

    for left, productions in grammar.items():
        valid = []

        for prod in productions:
            tokens = tokenize_by_nonterminals(prod, all_nonterminals)

            has_missing_nonterminal = any(
                tok in all_nonterminals and tok not in current_nonterminals
                for tok in tokens
            )

            if not has_missing_nonterminal:
                valid.append(prod)

        if valid:
            clean[left] = unique_preserve_order(valid)

    return clean


def create_new_start_symbol_if_epsilon(final_grammar, original_start, epsilon_nt):
    if original_start in epsilon_nt and original_start in final_grammar:
        new_start = original_start + "'"

        while new_start in final_grammar:
            new_start += "'"

        final_grammar[original_start] = [p for p in final_grammar[original_start] if p != ""]
        final_grammar[new_start] = [original_start]

        return final_grammar, new_start

    return final_grammar, original_start


def find_neperspektivne(grammar, non_terminals):
    productive = set()
    current_nonterminals = set(grammar.keys()) | set(non_terminals)
    changed = True

    while changed:
        changed = False

        for nt, productions in grammar.items():
            if nt in productive:
                continue

            for prod in productions:
                tokens = tokenize_by_nonterminals(prod, current_nonterminals)
                is_productive = True

                for sym in tokens:
                    if sym in current_nonterminals and sym not in productive:
                        is_productive = False
                        break

                if is_productive:
                    productive.add(nt)
                    changed = True
                    break

    return set(grammar.keys()) - productive


def remove_unproductive(grammar, unproductive):
    clean = {}
    current_nonterminals = set(grammar.keys())

    for nt, productions in grammar.items():
        if nt in unproductive:
            continue

        valid = []

        for prod in productions:
            tokens = tokenize_by_nonterminals(prod, current_nonterminals)

            if any(tok in unproductive for tok in tokens):
                continue

            valid.append(prod)

        if valid:
            clean[nt] = unique_preserve_order(valid)

    return clean


def find_unreachable(grammar, start_symbol, protected=None):
    if protected is None:
        protected = set()

    if start_symbol not in grammar:
        return set(grammar.keys()) - protected

    reachable = set(protected)
    reachable.add(start_symbol)
    queue = [start_symbol]
    current_nonterminals = set(grammar.keys())

    while queue:
        current = queue.pop()

        if current not in grammar:
            continue

        for prod in grammar[current]:
            tokens = tokenize_by_nonterminals(prod, current_nonterminals)

            for tok in tokens:
                if tok in current_nonterminals and tok not in reachable:
                    reachable.add(tok)
                    queue.append(tok)

    return set(grammar.keys()) - reachable


def remove_unreachable(grammar, unreachable, protected=None):
    if protected is None:
        protected = set()

    clean = {}
    current_nonterminals = set(grammar.keys())

    for nt, productions in grammar.items():
        if nt in unreachable and nt not in protected:
            continue

        valid = []

        for prod in productions:
            tokens = tokenize_by_nonterminals(prod, current_nonterminals)

            if any(tok in unreachable and tok not in protected for tok in tokens):
                continue

            valid.append(prod)

        if valid:
            clean[nt] = unique_preserve_order(valid)

    return clean


# =========================
# ODSTRÁNENIE ĽAVEJ REKURZIE
# =========================

def create_fresh_left_recursion_nonterminal(nt, grammar, ordered_nonterminals, orig_start):
    if nt == orig_start and "Z" not in grammar and "Z" not in ordered_nonterminals:
        candidate = "Z"
    else:
        candidate = nt + "'"

    while candidate in grammar or candidate in ordered_nonterminals:
        candidate += "'"

    return candidate


def remove_direct_left_recursion_single(nt, grammar, ordered_nonterminals, orig_start):
    if nt not in grammar:
        return None

    productions = unique_preserve_order(grammar[nt])
    current_nonterminals = set(grammar.keys()) | set(ordered_nonterminals)

    alpha = []
    beta = []

    for prod in productions:
        tokens = tokenize_by_nonterminals(prod, current_nonterminals)

        if tokens and tokens[0] == nt:
            alpha.append(join_tokens(tokens[1:]))
        else:
            beta.append(prod)

    if not alpha:
        grammar[nt] = productions
        return None

    new_nt = create_fresh_left_recursion_nonterminal(nt, grammar, ordered_nonterminals, orig_start)

    new_beta = []

    for item in beta:
        new_beta.append(item)
        new_beta.append(item + new_nt)

    new_alpha = []

    for item in alpha:
        if item == "":
            continue

        new_alpha.append(item)
        new_alpha.append(item + new_nt)

    grammar[nt] = unique_preserve_order(new_beta)
    grammar[new_nt] = unique_preserve_order(new_alpha)

    return new_nt


def remove_left_recursion(grammar, orig_start):
    grammar_copy = {left: unique_preserve_order(list(prods)) for left, prods in grammar.items()}
    ordered_nonterminals = list(grammar_copy.keys())

    i = 0

    while i < len(ordered_nonterminals):
        Ai = ordered_nonterminals[i]

        if Ai not in grammar_copy:
            i += 1
            continue

        for j in range(i):
            Aj = ordered_nonterminals[j]

            if Aj not in grammar_copy:
                continue

            current_nonterminals = set(grammar_copy.keys()) | set(ordered_nonterminals)
            new_prods = []

            for prod in grammar_copy[Ai]:
                tokens = tokenize_by_nonterminals(prod, current_nonterminals)

                if tokens and tokens[0] == Aj:
                    suffix = join_tokens(tokens[1:])

                    for gamma in grammar_copy[Aj]:
                        new_prods.append(gamma + suffix)
                else:
                    new_prods.append(prod)

            grammar_copy[Ai] = unique_preserve_order(new_prods)

        new_nt = remove_direct_left_recursion_single(
            Ai,
            grammar_copy,
            ordered_nonterminals,
            orig_start
        )

        if new_nt is not None and new_nt not in ordered_nonterminals:
            ordered_nonterminals.insert(i + 1, new_nt)

        i += 1

    return {left: prods for left, prods in grammar_copy.items() if prods}


# =========================
# GENEROVANIE REŤAZCOV
# =========================

def grammar_signature(grammar):
    return tuple(
        sorted(
            (lhs, tuple(sorted(set(prods))))
            for lhs, prods in grammar.items()
        )
    )


def make_exact_length_engine(grammar):
    if not grammar:
        def empty_exact(start_symbol, target_len):
            return frozenset({""}) if target_len == 0 else frozenset()

        return empty_exact

    nonterminals = set(grammar.keys())
    sorted_nts = sorted(nonterminals, key=len, reverse=True)

    def tokenize(prod):
        tokens = []
        i = 0

        while i < len(prod):
            matched = None

            for nt in sorted_nts:
                if prod.startswith(nt, i):
                    matched = nt
                    break

            if matched is not None:
                tokens.append(matched)
                i += len(matched)
            else:
                tokens.append(prod[i])
                i += 1

        return tuple(tokens)

    tokenized = {
        nt: [tokenize(prod) for prod in prods]
        for nt, prods in grammar.items()
    }

    INF = 10 ** 9
    min_len_nt = {nt: INF for nt in nonterminals}

    changed = True

    while changed:
        changed = False

        for nt, prods in tokenized.items():
            best = min_len_nt[nt]

            for tokens in prods:
                total = 0
                possible = True

                for tok in tokens:
                    if tok in nonterminals:
                        value = min_len_nt[tok]

                        if value == INF:
                            possible = False
                            break

                        total += value
                    else:
                        total += len(tok)

                if possible and total < best:
                    best = total

            if best < min_len_nt[nt]:
                min_len_nt[nt] = best
                changed = True

    @lru_cache(maxsize=None)
    def min_seq_len(tokens):
        total = 0

        for tok in tokens:
            if tok in nonterminals:
                value = min_len_nt[tok]

                if value == INF:
                    return INF

                total += value
            else:
                total += len(tok)

        return total

    in_progress_nt = set()

    @lru_cache(maxsize=None)
    def gen_nt_exact(nt, target_len):
        if nt not in tokenized:
            return frozenset()

        if target_len < 0 or min_len_nt.get(nt, INF) > target_len:
            return frozenset()

        key = (nt, target_len)

        if key in in_progress_nt:
            return frozenset()

        in_progress_nt.add(key)
        results = set()

        for tokens in tokenized[nt]:
            results.update(gen_seq_exact(tokens, target_len))

        in_progress_nt.remove(key)

        return frozenset(results)

    @lru_cache(maxsize=None)
    def gen_seq_exact(tokens, target_len):
        if target_len < 0:
            return frozenset()

        if not tokens:
            return frozenset({""}) if target_len == 0 else frozenset()

        if min_seq_len(tokens) > target_len:
            return frozenset()

        first = tokens[0]
        rest = tokens[1:]
        rest_min = min_seq_len(rest)
        results = set()

        if first in nonterminals:
            first_min = min_len_nt[first]
            max_first = target_len - rest_min

            for left_len in range(first_min, max_first + 1):
                left_set = gen_nt_exact(first, left_len)

                if not left_set:
                    continue

                right_set = gen_seq_exact(rest, target_len - left_len)

                if not right_set:
                    continue

                for left in left_set:
                    for right in right_set:
                        results.add(left + right)
        else:
            token_len = len(first)

            if target_len < token_len + rest_min:
                return frozenset()

            suffixes = gen_seq_exact(rest, target_len - token_len)

            for suffix in suffixes:
                results.add(first + suffix)

        return frozenset(results)

    return gen_nt_exact


def generate_strings_up_to_length(grammar, start_symbol, max_length):
    if not grammar or start_symbol not in grammar:
        return [""] if max_length >= 0 else []

    gen_nt_exact = make_exact_length_engine(grammar)
    all_strings = set()

    for length in range(max_length + 1):
        all_strings.update(gen_nt_exact(start_symbol, length))

    return sorted(all_strings, key=lambda x: (len(x), x))


def languages_equivalent_up_to_length(grammar1, start1, grammar2, start2, max_length):
    if start1 == start2 and grammar_signature(grammar1) == grammar_signature(grammar2):
        return True

    if not grammar1:
        gen1 = lambda s, length: frozenset({""}) if length == 0 else frozenset()
    else:
        gen1 = make_exact_length_engine(grammar1)

    if not grammar2:
        gen2 = lambda s, length: frozenset({""}) if length == 0 else frozenset()
    else:
        gen2 = make_exact_length_engine(grammar2)

    for length in range(max_length + 1):
        if gen1(start1, length) != gen2(start2, length):
            return False

    return True


def find_counterexample_up_to_length(grammar1, start1, grammar2, start2, max_length):
    if not grammar1:
        gen1 = lambda s, length: frozenset({""}) if length == 0 else frozenset()
    else:
        gen1 = make_exact_length_engine(grammar1)

    if not grammar2:
        gen2 = lambda s, length: frozenset({""}) if length == 0 else frozenset()
    else:
        gen2 = make_exact_length_engine(grammar2)

    for length in range(max_length + 1):
        set1 = gen1(start1, length)
        set2 = gen2(start2, length)

        only_in_g1 = sorted(set1 - set2)

        if only_in_g1:
            return only_in_g1[0], 1, 2

        only_in_g2 = sorted(set2 - set1)

        if only_in_g2:
            return only_in_g2[0], 2, 1

    return None


# =========================
# OPTIMALIZÁCIA GRAMATIKY
# =========================

def optimize_grammar(start_symbol, rules_input):
    original_grammar = process_rules(rules_input)
    original_non_terminals = list(original_grammar.keys())

    epsilon_nt = find_epsilon_producing(original_grammar, original_non_terminals)
    grammar_eps = remove_epsilon_productions(original_grammar, start_symbol, epsilon_nt)
    grammar_eps = remove_productions_with_missing_nonterminals(
        grammar_eps,
        original_non_terminals
    )

    grammar_with_start, new_start_symbol = create_new_start_symbol_if_epsilon(
        grammar_eps,
        start_symbol,
        epsilon_nt
    )

    grammar_no_simple = remove_simple_rules(grammar_with_start, find_simple_rules(grammar_with_start))
    grammar_left = remove_left_recursion(grammar_no_simple, start_symbol)

    grammar_no_simple = remove_simple_rules(grammar_left, find_simple_rules(grammar_left))
    grammar_left = remove_left_recursion(grammar_no_simple, start_symbol)

    protected = {new_start_symbol}

    unproductive = find_neperspektivne(grammar_left, list(grammar_left.keys()))
    grammar_prod = remove_unproductive(grammar_left, unproductive)

    unreachable = find_unreachable(grammar_prod, new_start_symbol, protected)
    grammar_reach = remove_unreachable(grammar_prod, unreachable, protected)

    final_grammar = merge_equivalent_non_terminals_fixpoint(
        grammar_reach,
        list(grammar_reach.keys())
    )

    return final_grammar, new_start_symbol


# =========================
# LINEARIZÁCIA GRAMATIKY
# =========================

def production_starts_with_nonterminal(prod, nonterminals):
    if prod == "":
        return False, None, prod

    tokens = tokenize_by_nonterminals(prod, nonterminals)

    if tokens and tokens[0] in nonterminals:
        first_nt = tokens[0]
        rest = join_tokens(tokens[1:])
        return True, first_nt, rest

    return False, None, prod


def is_terminal_prefixed_production(prod, nonterminals):
    if prod == "":
        return True

    starts_with_nt, _, _ = production_starts_with_nonterminal(prod, nonterminals)
    return not starts_with_nt


def is_linear_grammar(grammar):
    nonterminals = set(grammar.keys())

    for productions in grammar.values():
        for prod in productions:
            if not is_terminal_prefixed_production(prod, nonterminals):
                return False

    return True


def linearize_grammar(grammar, start_symbol, max_iterations=300):
    if not grammar:
        return {}, True

    current = {
        left: unique_preserve_order(list(productions))
        for left, productions in grammar.items()
    }

    success = False

    for _ in range(max_iterations):
        nonterminals = set(current.keys())
        changed = False
        new_grammar = {}

        for left, productions in current.items():
            new_productions = []

            for prod in productions:
                starts_with_nt, first_nt, rest = production_starts_with_nonterminal(
                    prod,
                    nonterminals
                )

                if starts_with_nt and first_nt in current:
                    for replacement in current[first_nt]:
                        new_productions.append(replacement + rest)
                    changed = True
                else:
                    new_productions.append(prod)

            new_grammar[left] = unique_preserve_order(new_productions)

        current = new_grammar

        current = remove_simple_rules(current, find_simple_rules(current))

        unproductive = find_neperspektivne(current, list(current.keys()))
        current = remove_unproductive(current, unproductive)

        unreachable = find_unreachable(current, start_symbol, protected={start_symbol})
        current = remove_unreachable(current, unreachable, protected={start_symbol})

        if is_linear_grammar(current):
            success = True
            break

        if not changed:
            break

    return current, success


# =========================
# VÝSLEDOK TESTOVANIA
# =========================

def remaining_rules_text(count):
    if count == 1:
        return "+ 1 ďalšie pravidlo"
    if 2 <= count <= 4:
        return f"+ {count} ďalšie pravidlá"
    return f"+ {count} ďalších pravidiel"


def count_total_productions(grammar):
    return sum(len(prods) for prods in grammar.values())


def format_rule_lines(lhs, productions, max_line_chars=58):
    display = ["ε" if p == "" else p for p in productions]

    if not display:
        return [f"  {lhs} ->"]

    lines = []
    first_prefix = f"  {lhs} -> "
    continuation_prefix = " " * len(first_prefix[:-1]) + "| "

    current = first_prefix

    for prod in display:
        separator = "" if current in (first_prefix, continuation_prefix) else " | "
        candidate = current + separator + prod

        if len(candidate) <= max_line_chars:
            current = candidate
        else:
            lines.append(current.rstrip())
            current = continuation_prefix + prod

    if current.strip():
        lines.append(current.rstrip())

    return lines


def configure_result_tags(text_widget=None):
    if text_widget is None:
        text_widget = result_text_output

    text_widget.tag_configure(
        "summary_title",
        font=("Arial", 16, "bold"),
        foreground=TEXT_COLOR,
        spacing1=4,
        spacing3=10
    )

    text_widget.tag_configure(
        "section_title",
        font=("Arial", 15, "bold"),
        foreground=TEXT_COLOR,
        spacing1=10,
        spacing3=6
    )

    text_widget.tag_configure(
        "submeta",
        font=("Arial", 12),
        foreground="#4a4a4a",
        spacing3=6
    )

    text_widget.tag_configure(
        "rule",
        font=("Consolas", 12),
        foreground=TEXT_COLOR,
        spacing1=1,
        spacing3=1
    )

    text_widget.tag_configure(
        "result_ok",
        font=("Arial", 14, "bold"),
        foreground="#0b6e2b",
        spacing1=6,
        spacing3=8
    )

    text_widget.tag_configure(
        "result_bad",
        font=("Arial", 14, "bold"),
        foreground="#b00020",
        spacing1=6,
        spacing3=4
    )

    text_widget.tag_configure(
        "counterexample",
        font=("Arial", 13, "italic"),
        foreground=TEXT_COLOR,
        spacing3=10
    )

    text_widget.tag_configure(
        "separator",
        foreground="#7a96ad"
    )


def prepare_preview_by_nonterminal(grammar, visible_limit=7):
    preview = []

    for lhs, prods in grammar.items():
        preview.append({
            "lhs": lhs,
            "shown": prods[:visible_limit],
            "hidden": prods[visible_limit:]
        })

    return preview


def build_result_payload(eq_length: int):
    start1 = g1_data.get("start")
    rules1 = g1_data.get("rules_lines", [])
    start2 = g2_data.get("start")
    rules2 = g2_data.get("rules_lines", [])

    lines = []
    both_rules_empty = (not rules1) and (not rules2)

    if not both_rules_empty:
        if not start1 or not rules1:
            lines.append("Gramatika G1 nie je úplne zadaná.")

        if not start2 or not rules2:
            lines.append("Gramatika G2 nie je úplne zadaná.")

        if lines:
            return {"plain_lines": lines}

    if both_rules_empty:
        final1, start1_opt = {}, start1 or ""
        final2, start2_opt = {}, start2 or ""
    else:
        final1, start1_opt = optimize_grammar(start1, rules1)
        final2, start2_opt = optimize_grammar(start2, rules2)

    equivalent = languages_equivalent_up_to_length(
        final1,
        start1_opt,
        final2,
        start2_opt,
        eq_length
    )

    counterexample = None

    if not equivalent:
        counterexample = find_counterexample_up_to_length(
            final1,
            start1_opt,
            final2,
            start2_opt,
            eq_length
        )

    return {
        "final1": final1,
        "final2": final2,
        "eq_length": eq_length,
        "equivalent": equivalent,
        "counterexample": counterexample
    }



def expand_rule_section(section_key):
    expanded_rule_sections.add(section_key)

    if last_result_payload is not None:
        render_result(last_result_payload)


def collapse_rule_section(section_key):
    expanded_rule_sections.discard(section_key)

    if last_result_payload is not None:
        render_result(last_result_payload)


def insert_clickable_result_text(text_widget, text, tag_name, callback):
    text_widget.insert(tk.END, text, (tag_name,))
    text_widget.tag_configure(
        tag_name,
        foreground=BUTTON_BG,
        underline=False,
        font=("Arial", 12, "bold")
    )
    text_widget.tag_bind(tag_name, "<Button-1>", lambda e, cb=callback: cb())
    text_widget.tag_bind(tag_name, "<Enter>", lambda e: text_widget.config(cursor="hand2"))
    text_widget.tag_bind(tag_name, "<Leave>", lambda e: text_widget.config(cursor="arrow"))


def insert_grammar_preview(text_widget, grammar_title, grammar, tag_prefix):
    text_widget.insert(tk.END, grammar_title + "\n", ("section_title",))

    if not grammar:
        text_widget.insert(
            tk.END,
            "Po optimalizácii gramatika ostala prázdna.\n\n",
            ("submeta",)
        )
        return

    nt_count = len(grammar)
    prod_count = count_total_productions(grammar)

    preview = prepare_preview_by_nonterminal(grammar, visible_limit=7)

    for i, item in enumerate(preview):
        lhs = item["lhs"]
        shown = item["shown"]
        hidden = item["hidden"]
        hidden_count = len(hidden)

        section_key = f"{tag_prefix}_{i}"

        if hidden_count == 0:
            for line in format_rule_lines(lhs, shown):
                text_widget.insert(tk.END, line + "\n", ("rule",))
            continue

        if section_key not in expanded_rule_sections:
            for line in format_rule_lines(lhs, shown):
                text_widget.insert(tk.END, line + "\n", ("rule",))

            text_widget.insert(tk.END, "      ")
            insert_clickable_result_text(
                text_widget,
                f"[Zobraziť {remaining_rules_text(hidden_count)}]",
                section_key,
                lambda key=section_key: expand_rule_section(key)
            )
            text_widget.insert(tk.END, "\n")
        else:
            all_rules = shown + hidden

            for line in format_rule_lines(lhs, all_rules):
                text_widget.insert(tk.END, line + "\n", ("rule",))

            text_widget.insert(tk.END, "      ")
            insert_clickable_result_text(
                text_widget,
                "[Zbaliť pravidlá]",
                section_key,
                lambda key=section_key: collapse_rule_section(key)
            )
            text_widget.insert(tk.END, "\n")

    text_widget.insert(tk.END, "\n")


def insert_linear_grammar_preview(text_widget, grammar_title, grammar, success, tag_prefix):
    if success:
        title = grammar_title
    else:
        title = grammar_title + " - nepodarilo sa úplne linearizovať"

    text_widget.insert(tk.END, title + "\n", ("section_title",))

    if not grammar:
        text_widget.insert(
            tk.END,
            "Lineárna gramatika je prázdna.\n\n",
            ("submeta",)
        )
        return

    nt_count = len(grammar)
    prod_count = count_total_productions(grammar)


    preview = prepare_preview_by_nonterminal(grammar, visible_limit=7)

    for i, item in enumerate(preview):
        lhs = item["lhs"]
        shown = item["shown"]
        hidden = item["hidden"]
        hidden_count = len(hidden)

        section_key = f"{tag_prefix}_{i}"

        if hidden_count == 0:
            for line in format_rule_lines(lhs, shown):
                text_widget.insert(tk.END, line + "\n", ("rule",))
            continue

        if section_key not in expanded_rule_sections:
            for line in format_rule_lines(lhs, shown):
                text_widget.insert(tk.END, line + "\n", ("rule",))

            text_widget.insert(tk.END, "      ")
            insert_clickable_result_text(
                text_widget,
                f"[Zobraziť {remaining_rules_text(hidden_count)}]",
                section_key,
                lambda key=section_key: expand_rule_section(key)
            )
            text_widget.insert(tk.END, "\n")
        else:
            all_rules = shown + hidden

            for line in format_rule_lines(lhs, all_rules):
                text_widget.insert(tk.END, line + "\n", ("rule",))

            text_widget.insert(tk.END, "      ")
            insert_clickable_result_text(
                text_widget,
                "[Zbaliť pravidlá]",
                section_key,
                lambda key=section_key: collapse_rule_section(key)
            )
            text_widget.insert(tk.END, "\n")

    text_widget.insert(tk.END, "\n")


def create_result_text_widget(parent, height=1):
    text_widget = tk.Text(
        parent,
        font=ENTRY_FONT,
        bg="white",
        fg=TEXT_COLOR,
        wrap="word",
        height=height,
        borderwidth=0,
        highlightthickness=0,
        state="normal",
        cursor="arrow",
        padx=0,
        pady=0
    )
    configure_result_tags(text_widget)

    def on_text_mousewheel(event):
        if "result_canvas" in globals() and result_canvas.bbox("all") is not None:
            result_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    text_widget.bind("<MouseWheel>", on_text_mousewheel)

    return text_widget


def finish_result_text_widget(text_widget):
    text_widget.update_idletasks()

    line_count = int(text_widget.index("end-1c").split(".")[0])
    text_widget.config(
        height=max(1, line_count + 1),
        state="disabled"
    )


def clear_result_content():
    for child in result_content_frame.winfo_children():
        child.destroy()


def add_summary_block(payload):
    summary_text = create_result_text_widget(result_content_frame, height=5)
    summary_text.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))

    if "plain_lines" in payload:
        summary_text.insert(tk.END, "Výstup\n", ("summary_title",))
        summary_text.insert(tk.END, "\n".join(payload["plain_lines"]), ("submeta",))
        finish_result_text_widget(summary_text)
        return

    summary_text.insert(tk.END, "Súhrn výsledku\n", ("summary_title",))
    summary_text.insert(
        tk.END,
        f"Testované do dĺžky: {payload['eq_length']}\n",
        ("submeta",)
    )

    if payload["equivalent"]:
        summary_text.insert(
            tk.END,
            "Výsledok: Jazyky generované gramatikami G1 a G2 sú ekvivalentné do zadanej dĺžky.\n",
            ("result_ok",)
        )
    else:
        summary_text.insert(
            tk.END,
            "Výsledok: Jazyky generované gramatikami G1 a G2 NIE sú ekvivalentné.\n",
            ("result_bad",)
        )

        counterexample = payload.get("counterexample")

        if counterexample is not None:
            word, belongs_to, not_belongs_to = counterexample
            word_text = "ε" if word == "" else f"'{word}'"

            summary_text.insert(
                tk.END,
                f"Protipríklad: slovo {word_text} patrí do G{belongs_to}, ale nepatrí do G{not_belongs_to}.\n",
                ("counterexample",)
            )

    finish_result_text_widget(summary_text)


def add_single_grammar_block(row_index, grammar_title, grammar, tag_prefix):
    block = tk.Frame(result_content_frame, bg="white")
    block.grid(row=row_index, column=0, sticky="nsew", padx=16, pady=(8, 10))
    block.grid_columnconfigure(0, weight=1)

    text_widget = create_result_text_widget(block)
    text_widget.pack(fill="both", expand=True)
    insert_grammar_preview(text_widget, grammar_title, grammar, tag_prefix)
    finish_result_text_widget(text_widget)


def add_result_separator(row_index):
    separator = tk.Label(
        result_content_frame,
        text="─" * 90,
        font=ENTRY_FONT,
        bg="white",
        fg="#7a96ad"
    )
    separator.grid(row=row_index, column=0, sticky="ew", padx=16, pady=(0, 4))


def render_result(payload):
    global last_result_payload

    last_result_payload = payload

    clear_result_content()

    result_content_frame.grid_columnconfigure(0, weight=1)

    add_summary_block(payload)

    if "plain_lines" in payload:
        result_update_scrollbar()
        return

    add_result_separator(1)

    add_single_grammar_block(
        2,
        "Optimalizovaná gramatika G1",
        payload["final1"],
        "more_g1"
    )

    add_result_separator(3)

    add_single_grammar_block(
        4,
        "Optimalizovaná gramatika G2",
        payload["final2"],
        "more_g2"
    )

    result_update_scrollbar()


# =========================
# GUI SETUP
# =========================

def setup_start_frame(frame):
    frame.grid_rowconfigure(0, weight=0)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=0)
    frame.grid_rowconfigure(3, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    tk.Label(
        frame,
        text="Testovanie ekvivalencie dvoch bezkontextových gramatík",
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=0, column=0, pady=(35, 10), sticky="n")

    btns = tk.Frame(frame, bg=BG_COLOR)
    btns.grid(row=2, column=0)

    create_button(
        btns,
        "Start",
        lambda: (reset_all_user_inputs(), show_frame(frame_input), g1_inputs["start"].focus_set()),
        width=18,
        height=2
    ).grid(row=0, column=0, padx=10, pady=(0, 10))

    create_button(
        btns,
        "Info",
        show_intro_popup,
        width=18,
        height=2
    ).grid(row=1, column=0, padx=10, pady=(0, 10))

    create_button(
        btns,
        "História",
        show_history_popup,
        width=18,
        height=2
    ).grid(row=2, column=0, padx=10)


def setup_input_frame(frame):
    global g1_inputs, g2_inputs, common_inputs

    frame.grid_rowconfigure(0, weight=0)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=0)
    frame.grid_columnconfigure(0, weight=1)

    title_row = tk.Frame(frame, bg=BG_COLOR)
    title_row.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="ew")
    title_row.grid_columnconfigure(0, weight=1)
    title_row.grid_columnconfigure(1, weight=0)
    title_row.grid_columnconfigure(2, weight=1)

    tk.Label(
        title_row,
        text="Zadávanie gramatík",
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=0, column=1)

    create_button(
        title_row,
        "Info",
        show_input_help_popup,
        width=8,
        height=1,
        font=("Arial", 12, "bold")
    ).grid(row=0, column=2, sticky="e")

    blocks = tk.Frame(frame, bg=BG_COLOR)
    blocks.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
    blocks.grid_rowconfigure(0, weight=1)
    blocks.grid_columnconfigure(0, weight=1)
    blocks.grid_columnconfigure(1, weight=1)

    g1_start, g1_rules = create_grammar_input_block(
        parent=blocks,
        title="Gramatika G1",
        column=0,
        padx=(0, 10)
    )

    g2_start, g2_rules = create_grammar_input_block(
        parent=blocks,
        title="Gramatika G2",
        column=1,
        padx=(10, 0)
    )

    g1_rules.bind("<FocusIn>", lambda e: set_active_rule_widget(g1_rules))
    g1_rules.bind("<Button-1>", lambda e: set_active_rule_widget(g1_rules))
    g2_rules.bind("<FocusIn>", lambda e: set_active_rule_widget(g2_rules))
    g2_rules.bind("<Button-1>", lambda e: set_active_rule_widget(g2_rules))

    g1_inputs = {"start": g1_start, "rules": g1_rules}
    g2_inputs = {"start": g2_start, "rules": g2_rules}

    set_active_rule_widget(g1_rules)

    entry_eq = create_input_controls(frame, g1_start, g1_rules, g2_start, g2_rules)
    common_inputs = {"eq_len": entry_eq}


def create_grammar_input_block(parent, title, column, padx):
    grammar_frame = tk.LabelFrame(
        parent,
        text=title,
        font=LABEL_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    )
    grammar_frame.grid(row=0, column=column, padx=padx, sticky="nsew")
    grammar_frame.grid_rowconfigure(1, weight=1)
    grammar_frame.grid_columnconfigure(1, weight=1)

    tk.Label(
        grammar_frame,
        text="S -",
        font=LABEL_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=0, column=0, pady=5, padx=10, sticky="w")

    start_entry = tk.Entry(grammar_frame, font=ENTRY_FONT)
    start_entry.grid(row=0, column=1, pady=5, padx=10, sticky="ew")

    tk.Label(
        grammar_frame,
        text="P -",
        font=LABEL_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=1, column=0, pady=5, padx=10, sticky="nw")

    rules_text = tk.Text(grammar_frame, font=ENTRY_FONT, height=8, wrap="word")
    rules_text.grid(row=1, column=1, pady=5, padx=10, sticky="nsew")

    return start_entry, rules_text


def create_input_controls(frame, g1_start, g1_rules, g2_start, g2_rules):
    controls = tk.Frame(frame, bg=BG_COLOR)
    controls.grid(row=2, column=0, pady=(0, 15), sticky="ew")
    controls.grid_columnconfigure(0, weight=1)
    controls.grid_columnconfigure(1, weight=0)
    controls.grid_columnconfigure(2, weight=1)

    middle = tk.Frame(controls, bg=BG_COLOR)
    middle.grid(row=0, column=1)

    symbol_bar = tk.Frame(middle, bg=BG_COLOR)
    symbol_bar.grid(row=0, column=0, columnspan=2, pady=(5, 10))

    for index, symbol in enumerate(["→", "->", "|", "()"]):
        create_button(
            symbol_bar,
            symbol,
            lambda s=symbol: insert_rule_symbol(s),
            width=5,
            height=1,
            font=("Arial", 12, "bold")
        ).grid(row=0, column=index, padx=5)

    tk.Label(
        middle,
        text="L_test -",
        font=LABEL_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=1, column=0, padx=(0, 10), pady=(5, 2), sticky="e")

    entry_eq = tk.Entry(middle, font=ENTRY_FONT, width=10)
    entry_eq.grid(row=1, column=1, padx=(0, 10), pady=(5, 2), sticky="w")

    btn_row = tk.Frame(middle, bg=BG_COLOR)
    btn_row.grid(row=2, column=0, columnspan=2, pady=(8, 5))

    create_button(
        btn_row,
        "Testovať ekvivalenciu",
        lambda: on_test(g1_start, g1_rules, g2_start, g2_rules, entry_eq),
        width=20,
        height=1
    ).grid(row=0, column=0, padx=10)

    create_button(
        btn_row,
        "Späť",
        lambda: show_frame(frame_start),
        width=10,
        height=1
    ).grid(row=0, column=1, padx=10)

    return entry_eq


def read_input_values(g1_start, g1_rules, g2_start, g2_rules):
    start1_raw = g1_start.get().strip()
    rules1_text = g1_rules.get("1.0", tk.END).strip()
    rules1_lines = rules1_text.split("\n") if rules1_text else []

    start2_raw = g2_start.get().strip()
    rules2_text = g2_rules.get("1.0", tk.END).strip()
    rules2_lines = rules2_text.split("\n") if rules2_text else []

    return start1_raw, rules1_text, rules1_lines, start2_raw, rules2_text, rules2_lines


def store_input_values(start1_raw, rules1_lines, start2_raw, rules2_lines):
    g1_data["start"] = normalize_start_symbol(start1_raw)
    g1_data["rules_lines"] = rules1_lines

    g2_data["start"] = normalize_start_symbol(start2_raw)
    g2_data["rules_lines"] = rules2_lines


def on_test(g1_start, g1_rules, g2_start, g2_rules, entry_eq):
    start1_raw, rules1_text, rules1_lines, start2_raw, rules2_text, rules2_lines = read_input_values(
        g1_start,
        g1_rules,
        g2_start,
        g2_rules
    )

    eq_length, eq_error = validate_and_parse_eq_length(entry_eq.get())

    store_input_values(start1_raw, rules1_lines, start2_raw, rules2_lines)

    if not rules1_text and not rules2_text:
        if eq_error:
            show_error_popup("Chyby vo vstupe", eq_error)
            return

        payload = build_result_payload(eq_length)
        save_result_to_json(payload)
        render_result(payload)
        show_frame(frame_result)
        return

    all_errors = validate_all_inputs_and_collect_errors(
        start1_raw,
        rules1_lines,
        start2_raw,
        rules2_lines
    )

    if eq_error:
        if all_errors:
            all_errors.append("")
        all_errors.append(eq_error)

    if all_errors:
        show_error_popup("Chyby vo vstupe", "\n".join(all_errors))
        return

    payload = build_result_payload(eq_length)
    save_result_to_json(payload)
    render_result(payload)
    show_frame(frame_result)


def setup_result_frame(frame):
    global result_content_frame, result_canvas, result_update_scrollbar

    frame.grid_rowconfigure(0, weight=0)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=0)
    frame.grid_columnconfigure(0, weight=1)

    tk.Label(
        frame,
        text="Výsledok",
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR
    ).grid(row=0, column=0, pady=(15, 5))

    output_frame = tk.Frame(frame, bg=BG_COLOR)
    output_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
    output_frame.grid_rowconfigure(0, weight=1)
    output_frame.grid_columnconfigure(0, weight=1)

    card = tk.Frame(
        output_frame,
        bg="white",
        bd=1,
        relief="solid"
    )
    card.grid(row=0, column=0, sticky="nsew")
    card.grid_rowconfigure(0, weight=1)
    card.grid_columnconfigure(0, weight=1)

    result_canvas = tk.Canvas(
        card,
        bg="white",
        borderwidth=0,
        highlightthickness=0
    )
    result_canvas.grid(row=0, column=0, sticky="nsew")

    scrollbar = tk.Scrollbar(card, orient="vertical", command=result_canvas.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    scrollbar.grid_remove()

    result_content_frame = tk.Frame(result_canvas, bg="white")
    content_window = result_canvas.create_window(
        (0, 0),
        window=result_content_frame,
        anchor="nw"
    )

    def update_scrollbar():
        result_canvas.update_idletasks()
        result_canvas.configure(scrollregion=result_canvas.bbox("all"))

        bbox = result_canvas.bbox("all")
        if bbox is None:
            scrollbar.grid_remove()
            return

        content_height = bbox[3] - bbox[1]

        if content_height > result_canvas.winfo_height():
            scrollbar.grid(row=0, column=1, sticky="ns")
        else:
            scrollbar.grid_remove()

    def yscrollcommand(*args):
        scrollbar.set(*args)

    result_canvas.configure(yscrollcommand=yscrollcommand)

    def on_content_configure(event):
        result_canvas.configure(scrollregion=result_canvas.bbox("all"))
        update_scrollbar()

    def on_canvas_configure(event):
        result_canvas.itemconfigure(content_window, width=event.width)
        update_scrollbar()

    result_content_frame.bind("<Configure>", on_content_configure)
    result_canvas.bind("<Configure>", on_canvas_configure)

    def on_mousewheel(event):
        if result_canvas.bbox("all") is not None:
            result_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    result_canvas.bind("<MouseWheel>", on_mousewheel)

    btns = tk.Frame(frame, bg=BG_COLOR)
    btns.grid(row=2, column=0, pady=(0, 15))

    create_button(
        btns,
        "Späť na zadávanie",
        lambda: (reset_all_user_inputs(), show_frame(frame_input), g1_inputs["start"].focus_set()),
        width=18,
        height=1
    ).grid(row=0, column=0, padx=10, pady=5)

    create_button(
        btns,
        "Na začiatok",
        lambda: (reset_all_user_inputs(), show_frame(frame_start)),
        width=14,
        height=1
    ).grid(row=0, column=1, padx=10, pady=5)

    result_update_scrollbar = update_scrollbar


# =========================
# HLAVNÉ OKNO
# =========================

root = tk.Tk()
root.title("Testovanie")
root.geometry("1250x720")
root.minsize(1100, 650)
root.configure(bg=BG_COLOR)

container = tk.Frame(root, bg=BG_COLOR)
container.pack(fill="both", expand=True)
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

frame_start = tk.Frame(container, bg=BG_COLOR)
frame_input = tk.Frame(container, bg=BG_COLOR)
frame_result = tk.Frame(container, bg=BG_COLOR)

for current_frame in (frame_start, frame_input, frame_result):
    current_frame.grid(row=0, column=0, sticky="nsew")

setup_start_frame(frame_start)
setup_input_frame(frame_input)
setup_result_frame(frame_result)

show_frame(frame_start)

root.mainloop()