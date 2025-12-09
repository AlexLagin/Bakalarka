import tkinter as tk
import itertools
import re

# Uloženie vstupov pre G1 a G2 medzi scénami
g1_data = {}
g2_data = {}


### VŠEOBECNÉ FUNKCIE ###

def show_frame(frame, clear_inputs=None):
    """Zobrazí daný rám a voliteľne vymaže vstupné polia."""
    if clear_inputs:
        for entry in clear_inputs:
            if isinstance(entry, tk.Entry):
                entry.delete(0, tk.END)
            elif isinstance(entry, tk.Text):
                entry.delete("1.0", tk.END)
    frame.tkraise()


### FUNKCIE PRE PRÁCU S GRAMATIKOU ###

def process_rules(rules_input):
    """
    Spracuje vstupné pravidlá a vráti ich ako slovník: neterminál -> zoznam produkcií.
    Každý riadok: S->aAB | b
    Pravidlo "()" sa interpretuje ako prázdny reťazec.
    """
    rules = {}
    for rule in rules_input:
        if "->" in rule:
            left, right = rule.split("->")
            left = left.strip()
            right = [r.strip() for r in right.split("|")]
            rules[left] = ["" if r == "()" else r for r in right]
    return rules


def find_simple_rules(grammar):
    """Nájde jednoduché pravidlá A -> B, kde B je neterminál (jedno veľké písmeno)."""
    simple_rules = {}
    for A, productions in grammar.items():
        for prod in productions:
            if len(prod) == 1 and prod.isupper():
                simple_rules.setdefault(A, []).append(prod)
    return simple_rules


def remove_simple_rules(grammar, simple_rules):
    """
    Odstráni jednoduché pravidlá A->B a pridá nové podľa pravidla:
    ak A->B a B->γ, tak A->γ.
    """
    new_grammar = {key: set(value) for key, value in grammar.items()}
    changed = True
    while changed:
        changed = False
        for A, B_list in simple_rules.items():
            for B in B_list:
                if B in grammar:
                    for prod in grammar[B]:
                        if prod not in new_grammar[A]:
                            new_grammar[A].add(prod)
                            changed = True
        simple_rules = find_simple_rules(new_grammar)

    for A in list(new_grammar.keys()):
        new_grammar[A] = {p for p in new_grammar[A]
                          if not (len(p) == 1 and p.isupper())}

    final_grammar = {A: list(v) for A, v in new_grammar.items()}
    return final_grammar


def canonical_form(prod):
    """Každý neterminál (veľké písmeno s voliteľným apostrofom) nahradí symbolom N."""
    return re.sub(r"[A-Z](')?", "N", prod)


def merge_equivalent_non_terminals_once(grammar, original_nonterminals):
    """Jedna iterácia zlúčenia ekvivalentných neterminálov podľa canonical formy."""
    reverse_grammar = {}
    for nt, productions in grammar.items():
        canon_prods = sorted(canonical_form(p) for p in productions)
        key = tuple(canon_prods)
        reverse_grammar.setdefault(key, []).append(nt)

    merged_grammar = dict(grammar)
    changed = False

    for nts in reverse_grammar.values():
        if len(nts) > 1:
            # preferujeme pôvodné neterminály
            candidates = [nt for nt in nts if nt in original_nonterminals]
            winner = candidates[0] if candidates else nts[0]
            for nt in nts:
                if nt == winner:
                    continue
                if nt in merged_grammar:
                    del merged_grammar[nt]
                for A in list(merged_grammar.keys()):
                    new_prods = []
                    for p in merged_grammar[A]:
                        new_prods.append(p.replace(nt, winner))
                    merged_grammar[A] = new_prods
            changed = True

    return merged_grammar, changed


def merge_equivalent_non_terminals_fixpoint(grammar, original_nonterminals):
    """Opakovane volá merge_equivalent_non_terminals_once, kým sa nedosiahne fixpoint."""
    changed = True
    current = grammar
    while changed:
        new_grammar, changed = merge_equivalent_non_terminals_once(
            current, original_nonterminals
        )
        current = new_grammar
    return current


def find_epsilon_producing(grammar, non_terminals):
    """Nájde všetky neterminály, ktoré môžu odvodiť prázdny reťazec (ε)."""
    epsilon_nt = set()
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
                else:
                    all_eps = True
                    for ch in prod:
                        if ch in grammar and ch not in epsilon_nt:
                            all_eps = False
                            break
                        elif ch not in grammar:
                            all_eps = False
                            break
                    if all_eps:
                        epsilon_nt.add(nt)
                        changed = True
                        break
    return epsilon_nt


def parse_production(prod):
    return list(prod)


def join_production(symbols):
    return "".join(symbols)


def remove_epsilon_productions(grammar, start_symbol, epsilon_nt):
    """
    Odstráni ε-pravidlá (A->ε) a vytvorí varianty produkcií,
    kde sa epsilonotvorné neterminály vynechajú.
    """
    new_grammar = {}
    for A in grammar.keys():
        new_grammar[A] = set()

    for A, productions in grammar.items():
        for p in productions:
            symbols = parse_production(p)
            nullable_positions = [i for i, sym in enumerate(symbols)
                                  if sym in epsilon_nt]

            subsets = itertools.chain.from_iterable(
                itertools.combinations(nullable_positions, r)
                for r in range(len(nullable_positions) + 1)
            )

            for subset in subsets:
                new_symbols = list(symbols)
                for idx in sorted(subset, reverse=True):
                    new_symbols.pop(idx)
                new_p = join_production(new_symbols)
                new_grammar[A].add(new_p)

    for A in list(new_grammar.keys()):
        if "" in new_grammar[A]:
            new_grammar[A].remove("")

    final_grammar = {}
    for A, prod_set in new_grammar.items():
        if prod_set:
            final_grammar[A] = list(prod_set)
    return final_grammar


def create_new_start_symbol_if_epsilon(final_grammar, original_start, epsilon_nt):
    """
    Ak je pôvodný štartovací symbol ε-tvorivý,
    pridá sa nový štartovací symbol S' s pravidlami:
       S' -> original_start  |  ε
    """
    if original_start in epsilon_nt and original_start in final_grammar:
        new_start = original_start + "'"
        final_grammar[new_start] = [original_start, ""]
        return final_grammar, new_start
    return final_grammar, original_start


def find_neperspektivne(grammar, non_terminals):
    """Zistí neperspektívne (neproduktívne) neterminály."""
    productive = set()
    changed = True
    while changed:
        changed = False
        for nt, productions in grammar.items():
            if nt in productive:
                continue
            for prod in productions:
                symbols = parse_production(prod)
                is_prod = True
                for sym in symbols:
                    if sym in non_terminals and sym not in productive:
                        is_prod = False
                        break
                if is_prod:
                    productive.add(nt)
                    changed = True
                    break
    return set(non_terminals) - productive


def remove_unproductive(grammar, unproductive):
    """Odstráni neperspektívne neterminály a pravidlá, ktoré ich obsahujú."""
    clean = {}
    for nt, productions in grammar.items():
        if nt in unproductive:
            continue
        valid = []
        for prod in productions:
            if any(u in prod for u in unproductive):
                continue
            valid.append(prod)
        if valid:
            clean[nt] = valid
    return clean


def find_unreachable(grammar, start_symbol, protected=set()):
    """Zistí nedostupné neterminály zo štartovacieho symbolu."""
    if start_symbol not in grammar:
        return set(grammar.keys()) - protected

    reachable = set(protected)
    reachable.add(start_symbol)
    queue = [start_symbol]

    while queue:
        cur = queue.pop()
        if cur not in grammar:
            continue
        for prod in grammar[cur]:
            for nt in grammar.keys():
                if nt in prod and nt not in reachable:
                    reachable.add(nt)
                    queue.append(nt)

    return set(grammar.keys()) - reachable


def remove_unreachable(grammar, unreachable, protected=set()):
    """
    Odstráni nedostupné neterminály a pravidlá, ktoré ich obsahujú,
    ale neterminály v 'protected' ponechá.
    """
    clean = {}
    for nt, productions in grammar.items():
        if nt in unreachable and nt not in protected:
            continue
        valid = []
        for prod in productions:
            if any(u in prod for u in unreachable if u not in protected):
                continue
            valid.append(prod)
        if valid:
            clean[nt] = valid
    return clean


### ĽAVÁ REKURZIA ###

def check_left_recursion(grammar):
    """
    Zisťuje ľavú rekurziu:
      - priama: A -> Aα
      - nepriama: A => Bα =>* Aα'
    """
    direct = set()
    indirect = set()

    for A, productions in grammar.items():
        for prod in productions:
            if not prod:
                continue
            # priama
            if prod.startswith(A):
                direct.add(A)
            # nepriama: A -> Bα a z B sa vieme ľavmostne dostať k A
            elif prod[0].isupper() and not prod.startswith(A):
                B = prod[0]
                if leads_leftmost_to_A(A, B, grammar):
                    indirect.add(A)

    return direct, indirect


def leads_leftmost_to_A(current, target, grammar, visited=None):
    if visited is None:
        visited = set()

    if current in visited:
        return False
    visited.add(current)

    for p in grammar.get(current, []):
        if not p:
            continue
        first_sym = p[0]
        if first_sym == target:
            return True
        if first_sym.isupper() and first_sym != target:
            if leads_leftmost_to_A(first_sym, target, grammar, visited.copy()):
                return True

    return False


def remove_direct_left_recursion_for(ordered_nonterminals, grammar, orig_start):
    """
    Odstráni priamu ľavú rekurziu pre dané neterminály.
    Ak ide o štartovací symbol, môže vytvoriť nový neterminál Z.
    """
    for nt in ordered_nonterminals:
        if nt not in grammar:
            continue
        prods = grammar[nt]
        alpha = []
        beta = []
        for prod in prods:
            if prod.startswith(nt):
                alpha.append(prod[len(nt):])
            else:
                beta.append(prod)

        if alpha:
            if nt == orig_start and 'Z' not in ordered_nonterminals:
                candidate = "Z"
                ordered_nonterminals.insert(0, candidate)
            else:
                candidate = nt + "'"

            while candidate in grammar:
                candidate += "'"

            grammar[candidate] = []
            new_beta = []
            for b in beta:
                new_beta.append(b)
                new_beta.append(b + candidate)
            grammar[nt] = new_beta

            new_alpha = []
            for a in alpha:
                new_alpha.append(a + candidate)
                new_alpha.append(a)
            grammar[candidate] = new_alpha


def remove_indirect_left_recursion_bottom_up(grammar, ordered_nonterminals,
                                             orig_start, direct):
    """
    Odstráni nepriamu ľavú rekurziu zdola nahor.
    """
    G = {A: list(prods) for A, prods in grammar.items()}

    if direct:
        remove_direct_left_recursion_for(ordered_nonterminals, G, orig_start)
        direct = False

    for i in reversed(range(len(ordered_nonterminals))):
        Ai = ordered_nonterminals[i]
        if Ai not in G:
            continue
        for j in range(i + 1):
            Aj = ordered_nonterminals[j]
            if Aj not in G:
                continue
            if Aj == Ai:
                continue
            new_prods = []
            for prod in G[Aj]:
                if prod.startswith(Ai):
                    alpha = prod[len(Ai):]
                    for gamma in G[Ai]:
                        new_prods.append(gamma + alpha)
                else:
                    new_prods.append(prod)
            G[Aj] = new_prods

    return G


def merge_new_with_original(grammar, original_nonterminals):
    """
    pre nové neterminály hľadá ekvivalentný pôvodný a premenováva ich.
    """
    new_grammar = dict(grammar)

    for nt in list(new_grammar.keys()):
        if nt in original_nonterminals:
            continue
        canon_nt = sorted(canonical_form(p) for p in new_grammar[nt])
        for old_nt in original_nonterminals:
            if old_nt in new_grammar:
                canon_old = sorted(canonical_form(p) for p in new_grammar[old_nt])
                if canon_nt == canon_old:
                    new_grammar = force_rename_new_to_old(new_grammar, old_nt, nt)
                    break

    return new_grammar


def merge_new_with_original_fixpoint(grammar, original_nonterminals):
    changed = True
    current = grammar
    while changed:
        new_grammar = merge_new_with_original(current, original_nonterminals)
        if new_grammar == current:
            changed = False
        else:
            current = new_grammar
            changed = True
    return current


def force_rename_new_to_old(grammar, old_nt, new_nt):
    """
    Vo všetkých produkciách nahradí výskyty new_nt za old_nt a new_nt odstráni.
    """
    if new_nt not in grammar or old_nt not in grammar:
        return grammar

    new_grammar = dict(grammar)

    for A in list(new_grammar.keys()):
        new_prods = []
        for p in new_grammar[A]:
            new_prods.append(p.replace(new_nt, old_nt))
        new_grammar[A] = new_prods

    if new_nt in new_grammar:
        del new_grammar[new_nt]

    return new_grammar


### GENEROVANIE REŤAZCOV ###

def generate_strings_up_to_length(grammar, start_symbol, max_length):
    """
    Vygeneruje všetky terminálne reťazce jazyka gramatiky `grammar`
    so štartom `start_symbol` a dĺžkou <= max_length.
    """
    if start_symbol not in grammar:
        return []

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
        return tokens

    memo_nt = {}
    memo_seq = {}

    def gen_nt(nt, remaining, stack):
        key = (nt, remaining)
        if key in memo_nt:
            return memo_nt[key]
        if key in stack:
            return set()
        stack.add(key)
        results = set()
        for prod in grammar.get(nt, []):
            if prod == "":
                if remaining >= 0:
                    results.add("")
                continue
            tokens = tokenize(prod)
            for s in gen_seq(tokens, remaining, stack):
                if len(s) <= remaining:
                    results.add(s)
        stack.remove(key)
        memo_nt[key] = results
        return results

    def gen_seq(tokens, remaining, stack):
        key = (tuple(tokens), remaining)
        if key in memo_seq:
            return memo_seq[key]
        if remaining < 0:
            return set()
        if not tokens:
            return {""}
        first, *rest = tokens
        results = set()
        if first in nonterminals:
            for s1 in gen_nt(first, remaining, stack):
                rem = remaining - len(s1)
                if rem < 0:
                    continue
                for s2 in gen_seq(rest, rem, stack):
                    results.add(s1 + s2)
        else:
            t = first
            if remaining < len(t):
                memo_seq[key] = set()
                return set()
            for s2 in gen_seq(rest, remaining - len(t), stack):
                results.add(t + s2)
        memo_seq[key] = results
        return results

    all_strings = set()
    for s in gen_nt(start_symbol, max_length, set()):
        if len(s) <= max_length:
            all_strings.add(s)

    return sorted(all_strings, key=lambda x: (len(x), x))


### HLAVNÁ FUNKCIA NA OPTIMALIZÁCIU GRAMATIKY ###

def optimize_grammar(start_symbol, rules_input):
    """
    Vstup: počiatočný symbol a zoznam riadkov pravidiel.
    Výstup: (final_grammar, new_start_symbol) po všetkých krokoch optimalizácie.
    """
    original_grammar = process_rules(rules_input)
    original_non_terminals = list(original_grammar.keys())

    epsilon_nt = find_epsilon_producing(original_grammar, original_non_terminals)
    grammar_eps = remove_epsilon_productions(original_grammar, start_symbol, epsilon_nt)

    grammar_with_start, new_start_symbol = create_new_start_symbol_if_epsilon(
        grammar_eps, start_symbol, epsilon_nt
    )

    grammar_no_simple = remove_simple_rules(
        grammar_with_start, find_simple_rules(grammar_with_start)
    )

    direct = False
    direct_rec, indirect_rec = check_left_recursion(grammar_no_simple)

    if indirect_rec:
        if direct_rec:
            direct = True
        ordered_nts = list(reversed(grammar_no_simple.keys()))
        grammar_left = remove_indirect_left_recursion_bottom_up(
            grammar_no_simple, ordered_nts, start_symbol, direct
        )
    else:
        ordered_nts = list(grammar_no_simple.keys())
        G = {A: list(prods) for A, prods in grammar_no_simple.items()}
        for A in ordered_nts:
            remove_direct_left_recursion_for([A], G, start_symbol)
        grammar_left = G
        direct_rec = set()
        direct = False

    grammar_no_simple = remove_simple_rules(grammar_left, find_simple_rules(grammar_left))
    direct_rec, indirect_rec = check_left_recursion(grammar_no_simple)

    if indirect_rec:
        ordered_nts = list(reversed(grammar_no_simple.keys()))
        if new_start_symbol != start_symbol and new_start_symbol not in ordered_nts:
            ordered_nts.insert(0, new_start_symbol)
        grammar_left = remove_indirect_left_recursion_bottom_up(
            grammar_no_simple, ordered_nts, start_symbol, direct
        )
    else:
        ordered_nts = list(grammar_no_simple.keys())
        G = {A: list(prods) for A, prods in grammar_no_simple.items()}
        for A in ordered_nts:
            remove_direct_left_recursion_for([A], G, start_symbol)
        grammar_left = G
        direct_rec = set()
        direct = False

    # neperspektívne a nedostupné
    protected = {new_start_symbol}
    unproductive = find_neperspektivne(grammar_left, original_non_terminals)
    grammar_prod = remove_unproductive(grammar_left, unproductive)
    unreachable = find_unreachable(grammar_prod, new_start_symbol, protected)
    grammar_reach = remove_unreachable(grammar_prod, unreachable, protected)

    final_grammar = merge_equivalent_non_terminals_fixpoint(
        grammar_reach, original_non_terminals
    )
    return final_grammar, new_start_symbol


def test_equivalence(eq_length, text_output, update_scrollbar):
    """
    Spraví optimalizáciu G1 a G2, vygeneruje reťazce do dĺžky eq_length
    a vypíše výsledok do text_output.
    """
    start1 = g1_data.get("start")
    rules1 = g1_data.get("rules_lines", [])
    start2 = g2_data.get("start")
    rules2 = g2_data.get("rules_lines", [])

    output_lines = []

    if not start1 or not rules1:
        output_lines.append("Gramatika G1 nie je úplne zadaná.")
    if not start2 or not rules2:
        output_lines.append("Gramatika G2 nie je úplne zadaná.")

    if output_lines:
        text_output.config(state="normal")
        text_output.delete("1.0", tk.END)
        text_output.insert("1.0", "\n".join(output_lines))
        text_output.config(state="disabled")
        update_scrollbar()
        return

    final1, start1_opt = optimize_grammar(start1, rules1)
    final2, start2_opt = optimize_grammar(start2, rules2)

    # Optimalizované gramatiky
    output_lines.append("Optimalizovaná gramatika G1:")
    if final1:
        for lhs, prods in final1.items():
            pstr = " | ".join("ε" if p == "" else p for p in prods)
            output_lines.append(f"  {lhs} -> {pstr}")
    else:
        output_lines.append("  Po optimalizácii je gramatika prázdna (žiadne pravidlá).")

    output_lines.append("")
    output_lines.append("Optimalizovaná gramatika G2:")
    if final2:
        for lhs, prods in final2.items():
            pstr = " | ".join("ε" if p == "" else p for p in prods)
            output_lines.append(f"  {lhs} -> {pstr}")
    else:
        output_lines.append("  Po optimalizácii je gramatika prázdna (žiadne pravidlá).")

    strings1 = []
    strings2 = []

    if eq_length is not None:
        output_lines.append("")
        output_lines.append(f"Reťazce jazyka G1 do dĺžky {eq_length}:")
        if final1:
            strings1 = generate_strings_up_to_length(final1, start1_opt, eq_length)
            if not strings1:
                strings1 = []
        else:
            # prázdna gramatika -> podľa zadania generuje len prázdne slovo
            strings1 = [""]

        s1 = ", ".join("()" if s == "" else s for s in strings1) or "(žiadne)"
        output_lines.append("  " + s1)

        output_lines.append("")
        output_lines.append(f"Reťazce jazyka G2 do dĺžky {eq_length}:")
        if final2:
            strings2 = generate_strings_up_to_length(final2, start2_opt, eq_length)
            if not strings2:
                strings2 = []
        else:
            strings2 = [""]

        s2 = ", ".join("()" if s == "" else s for s in strings2) or "(žiadne)"
        output_lines.append("  " + s2)

        set1 = set(strings1)
        set2 = set(strings2)

        output_lines.append("")
        if set1 == set2:
            output_lines.append(
                f"Výsledok: pre dĺžky ≤ {eq_length} sú jazyky G1 a G2 ekvivalentné."
            )
        else:
            output_lines.append(
                f"Výsledok: pre dĺžky ≤ {eq_length} NIE sú jazyky G1 a G2 ekvivalentné."
            )
            # už NEvypisujeme príklady len z G1/G2
    else:
        output_lines.append("")
        output_lines.append(
            "Testovacia dĺžka (L_test) nebola zadaná – reťazce jazyka sa negenerovali."
        )

    text_output.config(state="normal")
    text_output.delete("1.0", tk.END)
    text_output.insert("1.0", "\n".join(output_lines))
    text_output.config(state="disabled")
    update_scrollbar()


### GUI NASTAVENIA ###

BG_COLOR = '#d0e7f9'
TEXT_COLOR = '#00274d'
BUTTON_BG = '#00509e'
BUTTON_FG = 'white'
TITLE_FONT = ("Arial", 20, "bold")
LABEL_FONT = ("Arial", 14)
ENTRY_FONT = ("Arial", 14)
BUTTON_FONT = ("Arial", 16)


def setup_main_frame():
    frame_main.grid_rowconfigure(0, weight=0)
    frame_main.grid_rowconfigure(1, weight=1)
    frame_main.grid_columnconfigure(0, weight=1)

    title_frame = tk.Frame(frame_main, bg=BG_COLOR)
    title_frame.grid(row=0, column=0, pady=(40, 20), sticky="n")
    tk.Label(title_frame, text="Testovanie ekvivalencie", font=TITLE_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).pack()
    tk.Label(title_frame, text="dvoch bezkontextových gramatík", font=TITLE_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).pack()

    btn_frame = tk.Frame(frame_main, bg=BG_COLOR)
    btn_frame.grid(row=1, column=0, pady=(10, 0), sticky="n")
    tk.Button(
        btn_frame,
        text="Začať",
        command=lambda: show_frame(frame_grammar1),
        font=BUTTON_FONT,
        width=20,
        height=2,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
    ).pack(pady=5)


def setup_grammar1_frame(frame):
    frame.grid_rowconfigure(0, weight=0)
    frame.grid_rowconfigure(1, weight=0)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_rowconfigure(3, weight=0)
    frame.grid_columnconfigure(0, weight=1)

    tk.Label(frame, text="Zadávanie gramatiky G1", font=TITLE_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).grid(
        row=0, column=0, pady=(10, 5), sticky="n"
    )

    frame_inputs = tk.Frame(frame, bg=BG_COLOR)
    frame_inputs.grid(row=1, column=0, pady=(5, 10), sticky="n")

    # S - štart
    tk.Label(frame_inputs, text="S -", font=LABEL_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).grid(row=0, column=0, pady=5, sticky="w")
    entry_start = tk.Entry(frame_inputs, font=ENTRY_FONT)
    entry_start.grid(row=0, column=1, pady=5, padx=10, sticky="ew")

    # L - len informačne, nevyužíva sa v logike
    tk.Label(frame_inputs, text="L -", font=LABEL_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).grid(row=1, column=0, pady=5, sticky="w")
    entry_len = tk.Entry(frame_inputs, font=ENTRY_FONT)
    entry_len.grid(row=1, column=1, pady=5, padx=10, sticky="ew")

    # P - pravidlá
    tk.Label(frame_inputs, text="P -", font=LABEL_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).grid(row=2, column=0, pady=5, sticky="nw")
    entry_rules = tk.Text(frame_inputs, width=40, height=6, font=ENTRY_FONT)
    entry_rules.grid(row=2, column=1, pady=5, padx=10, sticky="ew")

    frame_inputs.grid_columnconfigure(1, weight=1)

    frame_buttons = tk.Frame(frame, bg=BG_COLOR)
    frame_buttons.grid(row=3, column=0, pady=(10, 0), sticky="n")

    def on_next():
        g1_data["start"] = entry_start.get().strip()
        rules_text = entry_rules.get("1.0", tk.END).strip()
        g1_data["rules_lines"] = rules_text.split("\n") if rules_text else []
        g1_data["length_hint"] = entry_len.get().strip()
        show_frame(frame_grammar2)

    tk.Button(
        frame_buttons,
        text="Ďalej (G2)",
        command=on_next,
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
    ).pack(pady=5)

    tk.Button(
        frame_buttons,
        text="Späť",
        command=lambda: show_frame(frame_main, [entry_start, entry_len, entry_rules]),
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
    ).pack(pady=5)


def setup_grammar2_frame(frame):
    frame.grid_rowconfigure(0, weight=0)
    frame.grid_rowconfigure(1, weight=0)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_rowconfigure(3, weight=0)
    frame.grid_columnconfigure(0, weight=1)

    tk.Label(frame, text="Zadávanie gramatiky G2", font=TITLE_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).grid(
        row=0, column=0, pady=(10, 5), sticky="n"
    )

    frame_inputs = tk.Frame(frame, bg=BG_COLOR)
    frame_inputs.grid(row=1, column=0, pady=(5, 10), sticky="n")

    tk.Label(frame_inputs, text="S -", font=LABEL_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).grid(row=0, column=0, pady=5, sticky="w")
    entry_start = tk.Entry(frame_inputs, font=ENTRY_FONT)
    entry_start.grid(row=0, column=1, pady=5, padx=10, sticky="ew")

    tk.Label(frame_inputs, text="L -", font=LABEL_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).grid(row=1, column=0, pady=5, sticky="w")
    entry_len = tk.Entry(frame_inputs, font=ENTRY_FONT)
    entry_len.grid(row=1, column=1, pady=5, padx=10, sticky="ew")

    tk.Label(frame_inputs, text="P -", font=LABEL_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).grid(row=2, column=0, pady=5, sticky="nw")
    entry_rules = tk.Text(frame_inputs, width=40, height=6, font=ENTRY_FONT)
    entry_rules.grid(row=2, column=1, pady=5, padx=10, sticky="ew")

    # L_test - dĺžka, do ktorej porovnávame jazyky
    tk.Label(frame_inputs, text="L_test -", font=LABEL_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).grid(row=3, column=0, pady=5, sticky="w")
    entry_eq_len = tk.Entry(frame_inputs, font=ENTRY_FONT)
    entry_eq_len.grid(row=3, column=1, pady=5, padx=10, sticky="ew")

    frame_inputs.grid_columnconfigure(1, weight=1)

    frame_buttons = tk.Frame(frame, bg=BG_COLOR)
    frame_buttons.grid(row=3, column=0, pady=(10, 0), sticky="n")

    def on_test():
        g2_data["start"] = entry_start.get().strip()
        rules_text = entry_rules.get("1.0", tk.END).strip()
        g2_data["rules_lines"] = rules_text.split("\n") if rules_text else []
        g2_data["length_hint"] = entry_len.get().strip()

        eq_text = entry_eq_len.get().strip()
        eq_length = None
        if eq_text:
            try:
                eq_length = int(eq_text)
                if eq_length < 0:
                    eq_length = None
            except ValueError:
                eq_length = None

        test_equivalence(eq_length, result_text_output, result_update_scrollbar)
        show_frame(frame_result)

    tk.Button(
        frame_buttons,
        text="Testovať ekvivalenciu",
        command=on_test,
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
    ).pack(pady=5)

    tk.Button(
        frame_buttons,
        text="Späť",
        command=lambda: show_frame(
            frame_grammar1, [entry_start, entry_len, entry_rules, entry_eq_len]
        ),
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
    ).pack(pady=5)


def setup_result_frame(frame):
    """Scéna s výsledkom testovania ekvivalencie."""
    global result_text_output, result_update_scrollbar

    frame.grid_rowconfigure(0, weight=0)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=0)
    frame.grid_columnconfigure(0, weight=1)

    tk.Label(frame, text="Výsledok testovania ekvivalencie", font=TITLE_FONT,
             bg=BG_COLOR, fg=TEXT_COLOR).grid(
        row=0, column=0, pady=(10, 5), sticky="n"
    )

    output_frame = tk.Frame(frame, bg=BG_COLOR)
    output_frame.grid(row=1, column=0, pady=10, padx=20, sticky="nsew")
    output_frame.grid_rowconfigure(0, weight=1)
    output_frame.grid_columnconfigure(0, weight=1)

    text_output = tk.Text(
        output_frame,
        font=ENTRY_FONT,
        bg=BG_COLOR,
        fg=TEXT_COLOR,
        wrap="word",
        height=12,
        borderwidth=0,
        highlightthickness=0,
    )
    text_output.grid(row=0, column=0, sticky="nsew")

    scrollbar = tk.Scrollbar(output_frame, orient="vertical",
                             command=text_output.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    scrollbar.grid_remove()

    def update_scrollbar():
        text_output.update_idletasks()
        info = text_output.dlineinfo("end-1c")
        if info is None:
            scrollbar.grid_remove()
            return
        y_bottom = info[1] + info[3]
        if y_bottom > text_output.winfo_height():
            scrollbar.grid(row=0, column=1, sticky="ns")
        else:
            scrollbar.grid_remove()

    def yscrollcommand(*args):
        scrollbar.set(*args)

    text_output.configure(yscrollcommand=yscrollcommand)

    def on_output_configure(event):
        update_scrollbar()

    output_frame.bind("<Configure>", on_output_configure)

    frame_buttons = tk.Frame(frame, bg=BG_COLOR)
    frame_buttons.grid(row=2, column=0, pady=(10, 10), sticky="n")

    tk.Button(
        frame_buttons,
        text="Späť na začiatok",
        command=lambda: show_frame(frame_main),
        font=BUTTON_FONT,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
    ).pack(pady=5)

    result_text_output = text_output
    result_update_scrollbar = update_scrollbar


### HLAVNÉ OKNO ###

root = tk.Tk()
root.title("Testovanie")
root.geometry("1000x650")
root.configure(bg=BG_COLOR)

container = tk.Frame(root)
container.pack(fill="both", expand=True)
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

frame_main = tk.Frame(container, bg=BG_COLOR)
frame_grammar1 = tk.Frame(container, bg=BG_COLOR)
frame_grammar2 = tk.Frame(container, bg=BG_COLOR)
frame_result = tk.Frame(container, bg=BG_COLOR)

for f in (frame_main, frame_grammar1, frame_grammar2, frame_result):
    f.grid(row=0, column=0, sticky="nsew")

setup_main_frame()
setup_result_frame(frame_result)
setup_grammar1_frame(frame_grammar1)
setup_grammar2_frame(frame_grammar2)

show_frame(frame_main)

root.mainloop()
