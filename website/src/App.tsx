import { lazy, Suspense } from "react";
import { SiteLayout } from "@/components/layout/SiteLayout";
import { useRouter } from "@/lib/router";

const HomePage = lazy(() => import("@/pages/HomePage"));
const PlayPage = lazy(() => import("@/pages/PlayPage"));
const GamesPage = lazy(() => import("@/pages/GamesPage"));
const LogbookPage = lazy(() => import("@/pages/LogbookPage"));

function RouteFallback() {
  return (
    <div className="mx-auto grid min-h-[60vh] max-w-[1180px] place-items-center px-6">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-text-tertiary">
        loading record
      </p>
    </div>
  );
}

export default function App() {
  const { pathname } = useRouter();
  const page = pathname === "/play"
    ? <PlayPage />
    : pathname === "/games"
    ? <GamesPage />
    : pathname === "/logbook"
      ? <LogbookPage />
      : <HomePage />;

  return (
    <Suspense fallback={<RouteFallback />}>
      <SiteLayout>{page}</SiteLayout>
    </Suspense>
  );
}
