/**
 * Catalog product picker for the AiChat input area.
 *
 * Provides a searchable popover for browsing products from the catalog API.
 */

import { useState, useRef, useEffect } from "react";
import { ChevronsUpDown, X } from "lucide-react";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Switch } from "./ui/switch";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "./ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";
import { fetchProducts, type ProductListItem } from "../services/api";

interface CatalogSearchProps {
  useCatalog: boolean;
  onUseCatalogChange: (v: boolean) => void;
  selectedProduct: ProductListItem | null;
  onSelectProduct: (p: ProductListItem | null) => void;
  disabled: boolean;
}

export function CatalogSearch({
  useCatalog,
  onUseCatalogChange,
  selectedProduct,
  onSelectProduct,
  disabled,
}: CatalogSearchProps) {
  const [catalogOpen, setCatalogOpen] = useState(false);
  const [catalogItems, setCatalogItems] = useState<ProductListItem[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [searchQ, setSearchQ] = useState("");
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!useCatalog) onSelectProduct(null);
  }, [useCatalog]);

  useEffect(() => {
    if (!catalogOpen) return;
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    searchDebounceRef.current = setTimeout(
      () => {
        setCatalogLoading(true);
        setCatalogError(null);
        fetchProducts({ q: searchQ || undefined, limit: 100 })
          .then((r) => {
            setCatalogItems(r.items);
            setCatalogError(null);
          })
          .catch((e: unknown) => {
            setCatalogItems([]);
            setCatalogError(e instanceof Error ? e.message : "Could not load products.");
          })
          .finally(() => setCatalogLoading(false));
      },
      searchQ.trim() ? 320 : 0,
    );
    return () => {
      if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    };
  }, [catalogOpen, searchQ]);

  return (
    <div className="glass-card px-3 py-2.5 mb-2 space-y-2 rounded-xl border border-border/40">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Switch
            id="use-catalog"
            checked={useCatalog}
            onCheckedChange={(v) => onUseCatalogChange(!!v)}
            disabled={disabled}
          />
          <Label htmlFor="use-catalog" className="text-sm font-medium cursor-pointer">
            Use catalog product
          </Label>
        </div>
      </div>
      {useCatalog && (
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <Popover
            open={catalogOpen}
            onOpenChange={(o) => {
              setCatalogOpen(o);
              if (o) {
                setSearchQ("");
                setCatalogError(null);
              }
            }}
          >
            <PopoverTrigger asChild>
              <Button
                type="button"
                variant="outline"
                role="combobox"
                aria-expanded={catalogOpen}
                className="min-w-[200px] max-w-full justify-between font-normal text-left h-9"
                disabled={disabled}
              >
                <span className="truncate">
                  {selectedProduct
                    ? `${selectedProduct.product_name.slice(0, 48)}${selectedProduct.product_name.length > 48 ? "…" : ""}`
                    : "Search products…"}
                </span>
                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[min(96vw,28rem)] p-0" align="start">
              <Command shouldFilter={false}>
                <CommandInput placeholder="Search by name…" value={searchQ} onValueChange={setSearchQ} />
                <CommandList>
                  {catalogLoading && (
                    <div className="py-6 text-center text-sm text-muted-foreground">Loading…</div>
                  )}
                  {!catalogLoading && catalogError && (
                    <div className="py-4 px-3 text-center text-sm text-destructive leading-snug" role="alert">
                      {catalogError}
                    </div>
                  )}
                  {!catalogLoading && !catalogError && catalogItems.length === 0 && (
                    <CommandEmpty>No products found.</CommandEmpty>
                  )}
                  {!catalogLoading && catalogItems.length > 0 && (
                    <CommandGroup>
                      {catalogItems.map((p) => (
                        <CommandItem
                          key={p.product_id}
                          value={p.product_id}
                          onSelect={() => {
                            onSelectProduct(p);
                            setCatalogOpen(false);
                          }}
                        >
                          <span className="flex flex-col gap-0.5 min-w-0">
                            <span className="truncate font-medium">{p.product_name}</span>
                            <span className="text-xs text-muted-foreground">
                              {p.price != null ? `$${p.price.toLocaleString()}` : "—"} · {p.product_id}
                            </span>
                          </span>
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  )}
                </CommandList>
              </Command>
            </PopoverContent>
          </Popover>
          {selectedProduct && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-9 w-9 shrink-0"
              onClick={() => onSelectProduct(null)}
              aria-label="Clear product"
            >
              <X className="h-4 w-4" />
            </Button>
          )}
          <p className="text-[11px] text-muted-foreground w-full basis-full">
            Default price band comes from the API (see <code className="text-[10px]">PRODUCT_BROWSE_PRICE_*</code>).
          </p>
        </div>
      )}
    </div>
  );
}
