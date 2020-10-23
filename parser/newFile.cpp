













using namespace Kripke;
using namespace Kripke::Core;



struct SweepSdom {

  static const std::string KernelName;

  template<typename AL>
  inline __attribute__((always_inline))
  void operator()(AL al, Kripke::Core::DataStore &data_store,
                  Kripke::SdomId sdom_id) const
  {

    using ExecPolicy = typename Kripke::Arch::Policy_SweepSubdomains<AL>::ExecPolicy;

    auto sdom_al = getSdomAL(al, sdom_id);

    int num_directions = data_store.getVariable<Set>("Set/Direction").size(sdom_id);
    int num_groups = data_store.getVariable<Set>("Set/Group").size(sdom_id);
    int local_imax = data_store.getVariable<Set>("Set/ZoneI").size(sdom_id);
    int local_jmax = data_store.getVariable<Set>("Set/ZoneJ").size(sdom_id);
    int local_kmax = data_store.getVariable<Set>("Set/ZoneK").size(sdom_id);

    auto xcos = sdom_al.getView(data_store.getVariable<Field_Direction2Double>("quadrature/xcos"));
    auto ycos = sdom_al.getView(data_store.getVariable<Field_Direction2Double>("quadrature/ycos"));
    auto zcos = sdom_al.getView(data_store.getVariable<Field_Direction2Double>("quadrature/zcos"));
    auto view_id = sdom_al.getView(data_store.getVariable<Field_Direction2Int>("quadrature/id"));
    auto view_jd = sdom_al.getView(data_store.getVariable<Field_Direction2Int>("quadrature/jd"));
    auto view_kd = sdom_al.getView(data_store.getVariable<Field_Direction2Int>("quadrature/kd"));

    auto dx = sdom_al.getView(data_store.getVariable<Field_ZoneI2Double>("dx"));
    auto dy = sdom_al.getView(data_store.getVariable<Field_ZoneJ2Double>("dy"));
    auto dz = sdom_al.getView(data_store.getVariable<Field_ZoneK2Double>("dz"));

    auto sigt = sdom_al.getView(data_store.getVariable<Kripke::Field_SigmaTZonal>("sigt_zonal"));
    auto psi = sdom_al.getView(data_store.getVariable<Kripke::Field_Flux>("psi"));
    auto rhs = sdom_al.getView(data_store.getVariable<Kripke::Field_Flux>("rhs"));

    auto psi_lf = sdom_al.getView(data_store.getVariable<Field_IPlane>("i_plane"));
    auto psi_fr = sdom_al.getView(data_store.getVariable<Field_JPlane>("j_plane"));
    auto psi_bo = sdom_al.getView(data_store.getVariable<Field_KPlane>("k_plane"));


    Direction d0{0};
    int id = view_id(d0);
    int jd = view_jd(d0);
    int kd = view_kd(d0);

    ZoneI start_i((id>0) ? 0 : local_imax-1);
    ZoneJ start_j((jd>0) ? 0 : local_jmax-1);
    ZoneK start_k((kd>0) ? 0 : local_kmax-1);

    ZoneI end_i((id>0) ? local_imax : -1);
    ZoneJ end_j((jd>0) ? local_jmax : -1);
    ZoneK end_k((kd>0) ? local_kmax : -1);

    auto zone_layout = data_store.getVariable<ProductSet<3>>("Set/Zone").getLayout(sdom_id);

    RAJA::kernel<ExecPolicy>(
        camp::make_tuple(
            RAJA::TypedRangeSegment<Direction>(0, num_directions),
            RAJA::TypedRangeSegment<Group>(0, num_groups),
            RAJA::TypedRangeStrideSegment<ZoneK>(*start_k, *end_k, kd),
            RAJA::TypedRangeStrideSegment<ZoneJ>(*start_j, *end_j, jd),
            RAJA::TypedRangeStrideSegment<ZoneI>(*start_i, *end_i, id)


    ),
        [=] __attribute__((host)) __attribute__((device)) (Direction d, Group g, ZoneK k, ZoneJ j, ZoneI i) {

            double xcos_dxi = 2.0 * xcos(d) / dx(i);
            double ycos_dyj = 2.0 * ycos(d) / dy(j);
            double zcos_dzk = 2.0 * zcos(d) / dz(k);

            Zone z(zone_layout(*i, *j, *k));


            double psi_d_g_z = (rhs(d,g,z)
                + psi_lf(d, g, j, k) * xcos_dxi
                + psi_fr(d, g, i, k) * ycos_dyj
                + psi_bo(d, g, i, j) * zcos_dzk)
                / (xcos_dxi + ycos_dyj + zcos_dzk + sigt(g, z));

            psi(d, g, z) = psi_d_g_z;


            psi_lf(d, g, j, k) = 2.0 * psi_d_g_z - psi_lf(d, g, j, k);
            psi_fr(d, g, i, k) = 2.0 * psi_d_g_z - psi_fr(d, g, i, k);
            psi_bo(d, g, i, j) = 2.0 * psi_d_g_z - psi_bo(d, g, i, j);

        }
    );
  }
};

const std::string SweepSdom::KernelName = "sweepSubdomain";

void Kripke::Kernel::sweepSubdomain(Kripke::Core::DataStore &data_store,
    Kripke::SdomId sdom_id)
{
  Kripke::BlockTimer BLK_TIMER_SweepSubdomain(data_store.getVariable<Kripke::Timing>("timing"), "SweepSubdomain");;

  ArchLayoutV al_v = data_store.getVariable<ArchLayout>("al").al_v;

  Kripke::dispatch(al_v, SweepSdom{}, data_store, sdom_id);

}
